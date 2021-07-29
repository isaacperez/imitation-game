import time
import math 

import tensorflow as tf

import dataset
import model
import CONST

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Module functions: LOSS
# -----------------------------------------------------------------------------
@tf.function
def loss_function(model_output, expected_output):
    """Given the model output and the expected output data, this function   
       calculate the model's loss.
    
    Args:
      - expected_output: one-hot tensor for the expected class idx. It has the shape: 
        [batch_size, len(CONST.CLASSES)]

      - model_output: raw tensor output of the model. It has the shape: 
        [batch_size, len(CONST.CLASSES)]
    """
    classification_loss = tf.nn.softmax_cross_entropy_with_logits(labels=expected_output, logits=model_output)

    return classification_loss



# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_loss_function():
    """Tests for the function loss_function"""
 
    # Build the dataset
    probabilities = dataset.build_probabilities_dict()
    path_input_data = dataset.get_train_paths()
    NUM_DATA = len(path_input_data)
    replications = [1 for _ in range(NUM_DATA)]
    
    BATCH_SIZE = 16
    train_dataset = dataset.build_train_dataset(batch_size=BATCH_SIZE, 
                                        num_batches_preloaded=5, 
                                        num_parallel_calls=4,
                                        allow_repetitions=False, 
                                        probabilities=probabilities,
                                        replications=replications,
                                        shuffle=False)

    # Build the model
    resnet50 = model.Model()

    # Test the dataset without repetitions
    count = 0
    for img, _, expected_output in train_dataset:       

        start = time.process_time()
        model_output = resnet50(img, training=False)
        individual_loss_vector = loss_function(model_output, expected_output)
        mean_loss = tf.reduce_mean(individual_loss_vector)
        end = time.process_time()

        assert mean_loss.ndim == 0 and mean_loss >= 0.0 and not math.isnan(mean_loss), \
            "mean_loss has not the expected value/shape: " + str(mean_loss)

        assert individual_loss_vector.ndim == 1, \
            "individual_loss_vector doesn't have the expected shape"

        assert individual_loss_vector.shape[0] == BATCH_SIZE, \
            "Batch size dimension has not the expected shape"

        assert tf.reduce_all(individual_loss_vector >= 0.0) \
            and not tf.reduce_any(tf.math.is_nan(individual_loss_vector)), \
            "individual_loss_vector has not the expected value: " + str(individual_loss_vector)

        assert mean_loss == tf.reduce_mean(individual_loss_vector), \
            "mean_loss != tf.reduce_mean(individual_loss_vector)"

        assert individual_loss_vector.dtype == tf.dtypes.float32, \
            "individual_loss_vector is not float32"

        assert mean_loss.dtype == tf.dtypes.float32, \
            "mean_loss is not float32"

        minimum = tf.reduce_min(individual_loss_vector)
        maximum = tf.reduce_max(individual_loss_vector)
        assert minimum != maximum, \
            "output values are equal"

        if count % 100 == 0:
            print("[test_loss_function()]:", count,
                " batch/es processed in " + str(end - start) + " seconds")

        count += 1


def do_tests():
    """Launch all test avaiable in this module"""
    test_loss_function()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Only launch all tests
    do_tests()


# -----------------------------------------------------------------------------
# Information
# -----------------------------------------------------------------------------
"""
    - This function always receives a 2-D tensor with shape [N, len(CONST.CLASSES)]
    - This function returns the mean cross-entropy loss and the vector for the individual loss
"""

