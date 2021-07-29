import tensorflow as tf
GPU_TO_USE = [int(0)]

if __name__ == "__main__":
    # Set the GPU to use the only one visible
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[gpu] for gpu in GPU_TO_USE], 'GPU')

    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)


import logging
import os
from time import gmtime, strftime
import glob 
import shutil
import signal
import json
import random 

import numpy as np 

import dataset
import loss
import metrics
import model
import CONST



# -----------------------------------------------------------------------------
# Hiperparams
# -----------------------------------------------------------------------------
LEARNING_RATE = 0.0001  #0.0001
OPTIMIZER = "sgd"  #"sgd"
BATCH_SIZE = int(4)
NUM_EPOCH = int(1000)
NUM_BATCHES_PRELOADED = tf.data.experimental.AUTOTUNE  #int(32)
NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE  #int(8)
SAVED_MODEL_PATH = None  # str(saved_model_path) or None if it's not used
THRESHOLD_BAD_CLASSIFICATION = 0.5

#DO_THE_BATCH_IN_N_PASSES = 1  # > 0 


L2_PENALTY = 0.0#0.0001

MAX_REPLICATION = 1
MIN_REPLICATION = 1
GOOD_CLASSIFICATION_FACTOR = MAX_REPLICATION
BAD_CLASSIFICATION_FACTOR = 2
MAX_PATHS_TO_SHOW = 10


if __name__ == "__main__":

    if OPTIMIZER == "adam":
        OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "sgd":
        OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, clipnorm=1.0)
    else:
        print("Error. Unknown optimizer name")
        exit()


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Where the training results folder are going to be build
BASE_PATH = "../modelos/"

# Directory name where all the results are going to be saved in the output directory
RESULTS_DIR_NAME = "results"

# Directory name where CSV dataset files are going to be saved in the output directory
DATASET_DIR_NAME = "dataset"

# Directory name where the notebooks files are going to be saved in the output directory
NOTEBOOK_DIR_NAME = "notebooks"

# Name for the log file
LOG_FILE_NAME = "log.log"

# Make the logger
logger = logging.getLogger()
logger.setLevel('INFO')

# create console handler and set level to debug
ch = logging.StreamHandler()

ch.setLevel('INFO')

# create formatter
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s',
                              '%d-%m-%y %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(ch)

# Code to interrupt training with Ctrl + C
INTERRUPT_TRAINING = False

def signal_handler(sig, frame):
    logger.info("Ctrl+C pressed. Stopping training...'")
    global INTERRUPT_TRAINING
    INTERRUPT_TRAINING = True

# Calculate the loss for the threshold indicated
THRESHOLD_BAD_CLASSIFICATION_LOSS = -np.log(THRESHOLD_BAD_CLASSIFICATION) 

# -----------------------------------------------------------------------------
# Module functions: UTILS
# -----------------------------------------------------------------------------
def createOutputDirAndFillWithInitialCode():
    """Make a directory with the current time as name and return the path"""
    ouput_directory_path = strftime("%Y%m%d_%H%M", gmtime())

    if not os.path.exists(BASE_PATH + ouput_directory_path):
        
        # Create the directories
        os.makedirs(BASE_PATH + ouput_directory_path)
        os.makedirs(BASE_PATH + ouput_directory_path + os.sep + RESULTS_DIR_NAME)
        output_directory_path = BASE_PATH + ouput_directory_path + os.sep + RESULTS_DIR_NAME + os.sep

        # Create the log file
        fh = logging.FileHandler(BASE_PATH + ouput_directory_path
                                 + os.sep + LOG_FILE_NAME)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Copy all python files
        python_files_to_copy = \
        glob.glob(__file__[:-len(os.path.basename(__file__))] + "*.py")

        for file in python_files_to_copy:
            shutil.copy(file, BASE_PATH + ouput_directory_path + os.sep)

        # Copy dataset files
        os.makedirs(BASE_PATH + ouput_directory_path + os.sep + DATASET_DIR_NAME + os.sep)
        for file in glob.glob(dataset.DATASET_PATH + "*.csv"):
            shutil.copy(file, BASE_PATH + ouput_directory_path + os.sep + DATASET_DIR_NAME + os.sep)

        # Copy notebook
        os.makedirs(BASE_PATH + ouput_directory_path + os.sep + NOTEBOOK_DIR_NAME + os.sep)
        for file in glob.glob(__file__[:-len(os.path.basename(__file__))] + NOTEBOOK_DIR_NAME + os.sep + "*"):
            if os.path.isfile(file):
                shutil.copy(file, BASE_PATH + ouput_directory_path + os.sep + NOTEBOOK_DIR_NAME + os.sep)

        return output_directory_path

    else:
        logger.error("Output directory: " + BASE_PATH
                     + ouput_directory_path + " already exists")

        exit()


def updateBestEvaluationResultsAndSaveTheModel(evaluation_results, best_evaluation_results, model, output_directory_path, current_epoch):
    """Check whether the current evaluation results improves the best evaluation results"""

    for key in best_evaluation_results.keys():
        
        if "loss" in key and evaluation_results[key] <= best_evaluation_results[key]:
            logger.info((key + " has been improved: ").ljust(38) + str(evaluation_results[key]).ljust(6) + " < " + str(best_evaluation_results[key]).ljust(6))

            best_evaluation_results[key] = evaluation_results[key]
            
            # Save the model
            model.save_weights(output_directory_path + key)

        elif "loss" not in key and evaluation_results[key] >= best_evaluation_results[key]:
            logger.info((key + " has been improved: ").ljust(38) + str(evaluation_results[key]).ljust(6) + " > " + str(best_evaluation_results[key]).ljust(6))
            best_evaluation_results[key] = evaluation_results[key]
 
            # Save the model
            model.save_weights(output_directory_path + key)

    current_epoch_results = {}
    for key in evaluation_results.keys():
        current_epoch_results[key] = str(evaluation_results[key])

    # Save the results for the current epoch
    with open(output_directory_path + str(current_epoch) + '.json', 'w') as f:
        json.dump(current_epoch_results, f)
        
    return best_evaluation_results


def updateReplications(individual_training_loss, idx_src, replications):
    """Update the replications list by the current loss"""
    for i in range(BATCH_SIZE):
            if (individual_training_loss[i] > THRESHOLD_BAD_CLASSIFICATION_LOSS):
                replications[idx_src[i]] =  min(MAX_REPLICATION, replications[idx_src[i]] + BAD_CLASSIFICATION_FACTOR)
            else:
                replications[idx_src[i]] =  max(MIN_REPLICATION, replications[idx_src[i]] - GOOD_CLASSIFICATION_FACTOR)

    return replications


def showWorseImagesByReplications(path_input_data, replications):
    """Show the paths of the worse images which are reach the maximum replication factor"""

    indices = [index for index,value in enumerate(replications) if value >= MAX_REPLICATION]
    num_images_to_show = min(len(indices), MAX_PATHS_TO_SHOW)

    indicesSample = np.sort(random.sample(indices, num_images_to_show))

    for idx in indicesSample:
        logger.info("Path dificult image: " + path_input_data[idx])

    if len(indicesSample) < len(indices):
        logger.info("We have reach the maximum number of paths to show, stopping...")


def evaluateModel(dataset, model):
    """Calculate the loss and all the metrics for the model over the dataset"""
    
    logger.info("Begining evaluation...")

    # Loop over the dataset
    metrics_handler = metrics.Metrics()
    idx_batch = 0
    for img, _, expected_output in dataset:

        # Get results
        model_output = model(img, training=tf.constant(False))

        # Get the loss
        individual_loss = loss.loss_function(model_output, expected_output)

        # Update results
        metrics_handler.updateStatus(tf.argmax(model_output, -1), 
                                     tf.argmax(expected_output, -1), individual_loss)

        logger.info(str(int(idx_batch)) + " batches processed")
        idx_batch += 1

    # Calculate the final evaluation values
    evaluation_results = metrics_handler.calculateMetrics()

    return evaluation_results


# -----------------------------------------------------------------------------
# Module functions: TRAIN
# -----------------------------------------------------------------------------
def train():
    """Launch the training with the hiperparams defined"""

    # Prepare the output dir
    output_directory_path = createOutputDirAndFillWithInitialCode()

    # Get the dataset variables for training
    probabilities, \
        path_input_data, \
        replications, \
        batches_in_the_train_dataset = dataset.get_basic_training_variables(batch_size=BATCH_SIZE)

    # Build the datasets
    train_dataset = dataset.build_train_dataset(batch_size=BATCH_SIZE, 
                                                num_batches_preloaded=NUM_BATCHES_PRELOADED, 
                                                num_parallel_calls=NUM_PARALLEL_CALLS,
                                                allow_repetitions=False, 
                                                probabilities=probabilities,
                                                replications=replications,
                                                shuffle=True)

    val_dataset = dataset.build_val_dataset(batch_size=int(BATCH_SIZE),
                                            num_batches_preloaded=NUM_BATCHES_PRELOADED, 
                                            num_parallel_calls=NUM_PARALLEL_CALLS)

    # Built the model
    current_model = model.Model(l2_penalty=L2_PENALTY)
    current_model.build((None, CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT))
    print(current_model.summary())

    # Continue training
    if SAVED_MODEL_PATH:
        current_model.load_weights(SAVED_MODEL_PATH)

    # First evaluation
    best_evaluation_results = metrics.DEFAULT_EVALUATION_RESULTS

    # Begining training
    logger.info("Begining training...")

    """
    # Built the train function
    @tf.function
    def train_step_n_passes(img, expected_output):

        # Split the data
        imgs= tf.split(img, num_or_size_splits=DO_THE_BATCH_IN_N_PASSES)
        expected_outputs = tf.split(expected_output, num_or_size_splits=DO_THE_BATCH_IN_N_PASSES)

        # Fist pass is always done
        img = imgs[0]
        expected_output = expected_outputs[0]

        # First foward pass
        with tf.GradientTape() as tape:
            model_output = current_model(img, training=tf.constant(True))

            individual_training_loss = loss.loss_function(model_output, expected_output)
            mean_training_loss = tf.reduce_sum(individual_training_loss) * (1.0 / (BATCH_SIZE / DO_THE_BATCH_IN_N_PASSES))
            regularization_loss = tf.add_n(current_model.losses)

        # Calculate the gradients
        grads = tape.gradient(mean_training_loss, current_model.trainable_variables)

        all_grads = [grad / DO_THE_BATCH_IN_N_PASSES for grad in grads]
        all_individual_training_loss = individual_training_loss

        # Do the rest passes
        for i in range(1, DO_THE_BATCH_IN_N_PASSES):
            
            # Get the current data
            img = imgs[i]
            expected_output = expected_outputs[i]

            # First foward pass
            with tf.GradientTape() as tape:
                model_output = current_model(img, training=tf.constant(True))

                individual_training_loss = loss.loss_function(model_output, expected_output)
                mean_training_loss = tf.reduce_sum(individual_training_loss) * (1.0 / (BATCH_SIZE / DO_THE_BATCH_IN_N_PASSES))

            # Calculate the gradients
            grads = tape.gradient(mean_training_loss, current_model.trainable_variables)

            # Save the current individual training losses and gradients
            all_individual_training_loss = tf.concat([all_individual_training_loss, individual_training_loss], axis=0)
            for i in range(len(all_grads)):
                all_grads[i] = all_grads[i] + grads[i] / DO_THE_BATCH_IN_N_PASSES

        # Do the backward pass and adjust weights
        OPTIMIZER.apply_gradients(zip(all_grads, current_model.trainable_variables))

        return all_individual_training_loss, regularization_loss
    """
    @tf.function
    def train_step(img, expected_output):

        with tf.GradientTape() as tape:

            # Get the output of the model
            model_output = current_model(img, training=tf.constant(True))

            individual_training_loss = loss.loss_function(model_output, expected_output)
            mean_training_loss = tf.reduce_sum(individual_training_loss) * (1.0 / BATCH_SIZE)

            regularization_loss = tf.add_n(current_model.losses)

            mean_training_loss = mean_training_loss + regularization_loss

        # Calculate the gradients
        grads = tape.gradient(mean_training_loss, current_model.trainable_variables)
        #grads = [tf.where(tf.abs(x) < 0.001, x*0.001, tf.where(tf.abs(x) < 0.1, x*0.0001, x*0.00001)) for x in grads]

        # Do the backward pass and adjust weights
        OPTIMIZER.apply_gradients(zip(grads, current_model.trainable_variables))

        return individual_training_loss, regularization_loss

    # Training loop
    for epoch in range(NUM_EPOCH):

        it_train = 0
        accumulated_loss = 0.0

        for img, idx_src, expected_output in train_dataset:

            # Finish the program if we have reach the maximum or
            # Ctr+C signal detected
            if INTERRUPT_TRAINING:
                exit()

            # Do the train step
            individual_training_loss, regularization_loss = train_step(img, expected_output)

            # Save the current results
            accumulated_loss += tf.reduce_sum(individual_training_loss)

            # Update replications list
            replications = updateReplications(individual_training_loss, idx_src, replications)
            
            # Show the results of the current iteration
            logger.info("[EPOCH " + str(epoch).ljust(6) + " / "
                        + str(NUM_EPOCH) + "][TRAIN It: "
                        + str(it_train).ljust(6) + " / " + str(batches_in_the_train_dataset)  +"]: "
                        + str(np.round(tf.reduce_mean(individual_training_loss), 4)).ljust(6)
                        + " - Regularization loss = " + str(np.round(regularization_loss, 4)).ljust(6)) 

            it_train += 1

        # Do the validation after the training dataset
        logger.info("[EPOCH " + str(epoch).ljust(6) + " of "
                    + str(NUM_EPOCH) + "][VALIDATION]")

        # Validate the model
        evaluation_results = evaluateModel(val_dataset, current_model)

        # Save the training loss
        evaluation_results[CONST.MEAN_LOSS_TRAIN] = np.round(accumulated_loss.numpy() / float(it_train * BATCH_SIZE), 4)

        # Shot the results for the current epoch
        for key in best_evaluation_results.keys():
            logger.info("[EPOCH " + str(epoch).ljust(6) + " of "
                    + str(NUM_EPOCH) + "] " \
                    + (key + ": ").ljust(18) + str(evaluation_results[key]).ljust(6))

        # Show the worst images
        if epoch > 20:
            showWorseImagesByReplications(path_input_data, replications)

        # Look for improvements
        best_evaluation_results  = updateBestEvaluationResultsAndSaveTheModel(
            evaluation_results=evaluation_results, 
            best_evaluation_results=best_evaluation_results, 
            model=current_model, 
            output_directory_path=output_directory_path, 
            current_epoch=epoch+1)

        # Prepare a new dataset for training
        train_dataset = dataset.build_train_dataset(batch_size=BATCH_SIZE, 
                                                    num_batches_preloaded=NUM_BATCHES_PRELOADED, 
                                                    num_parallel_calls=NUM_PARALLEL_CALLS,
                                                    allow_repetitions=False, 
                                                    probabilities=probabilities,
                                                    replications=replications,
                                                    shuffle=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Redirect Ctrl + C signal
    signal.signal(signal.SIGINT, signal_handler)

    # Begin the training
    train()


# -----------------------------------------------------------------------------
# Information
# -----------------------------------------------------------------------------
"""
    - 
"""
