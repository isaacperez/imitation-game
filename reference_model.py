import time

import tensorflow as tf

import dataset
import CONST

import train

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

INITIAL_FILTERS_NAME = 16

# -----------------------------------------------------------------------------
# Module functions: MODEL
# -----------------------------------------------------------------------------
class Model(tf.keras.Model):
    
    def __init__(self, l2_penalty=0.0):
        super(Model, self).__init__(name='EfficientNetB5')

        # ----------------------------------------------------------------------
        # Internal constants
        # ----------------------------------------------------------------------
        self.num_classes = len(CONST.CLASSES)
        self.l2_penalty = l2_penalty
        self.initializer = tf.keras.initializers.glorot_normal

        # ----------------------------------------------------------------------
        # Model definition
        # ----------------------------------------------------------------------
        self.base_model = tf.keras.applications.EfficientNetB5(
            input_shape=[CONST.HIGH_SIZE, CONST.WIDTH_SIZE, CONST.NUM_CHANNELS_INPUT],
            include_top=False,
            weights='imagenet')
        self.base_model.trainable = True

        # Final part
        self.GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(units=self.num_classes, use_bias=False,
                                                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_penalty), 
                                                        bias_regularizer=tf.keras.regularizers.l2(self.l2_penalty), 
                                                        activation=None)

    def head(self, features, training=tf.constant(False)):
        """Process the features from extract_features() with the head of the network"""
        return self.prediction_layer(self.GlobalAveragePooling2D(features), training=training)

    def extract_features(self, inputs, training=tf.constant(False)):
        """This fuction acts as a wrapper for the base model freezing it"""

        return self.base_model(inputs, training)

    def call(self, inputs, training=tf.constant(False)):
        return self.head(self.extract_features(inputs, training=training), training=training)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_model():
    """Tests for the class ResNet50"""
 
    # Build the dataset
    probabilities = dataset.build_probabilities_dict()
    path_input_data, _, _ = dataset.get_train_PIC_elements()
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

    # Model
    model = Model()

    # Test the dataset without repetitions
    count = 0
    for img, _, _ in train_dataset:       

        start = time.process_time()
        model_output = model(img, training=False)
        end = time.process_time()

        assert model_output.ndim == 2 and model_output.shape[1] == len(CONST.CLASSES), \
            "model_output doesn't have the expected shape"
        
        assert model_output.shape[0] == BATCH_SIZE, \
            "Batch size dimension has not the expected shape"

        assert model_output.dtype == tf.dtypes.float32, \
            "model_output is not float32"

        minimum = tf.reduce_min(model_output)
        maximum = tf.reduce_max(model_output)
        assert minimum != maximum, \
            "output values are equal"

        if count % 100 == 0:
            print("[test_model()]:", count,
                " batch/es processed in " + str(end - start) + " seconds")

        count += 1


def do_tests():
    """Launch all test avaiable in this module"""
    test_model()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Only launch all tests
    model = Model()
    print(model.base_model.summary())


# -----------------------------------------------------------------------------
# Information
# -----------------------------------------------------------------------------
"""
    - This model always receives a 4-D tensor with shape [NHWC]
"""
