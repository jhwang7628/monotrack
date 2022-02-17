from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import layers as keras_layers

from tensorflow.python.layers import base

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn


class RandomizedReLUKeras(keras_layers.Layer):
    def __init__(self, a=3, b=8, seed=None, **kwargs):
        super(RandomizedReLUKeras, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.seed = seed

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs_training():

            with ops.name_scope("random_relu_training"):
                x = ops.convert_to_tensor(inputs, name="x")
                if not x.dtype.is_floating:
                    raise ValueError("x has to be a floating point tensor since it's going to"
                                     " be scaled. Got a %s tensor instead." % x.dtype)
                if isinstance(self.a, numbers.Real) and not 0 < self.a:
                    raise ValueError("a must be a scalar tensor or a float positive "
                                     ", got %g" % self.a)

                if isinstance(self.b, numbers.Real) and not 0 < self.b:
                    raise ValueError("b must be a scalar tensor or a float positive "
                                     ", got %g" % self.b)

                if isinstance(self.b, numbers.Real) and \
                        isinstance(self.a, numbers.Real) and \
                        not self.a < self.b:
                    raise ValueError("a and b must be a scalar tensor or a float such that"
                                     " a < b, got {} and {}".format(self.a, self.b))

                else:
                    a = ops.convert_to_tensor(self.a, dtype=x.dtype, name="a")
                    b = ops.convert_to_tensor(self.b, dtype=x.dtype, name="b")
                    a.get_shape().assert_is_compatible_with(tensor_shape.TensorShape([]))
                    b.get_shape().assert_is_compatible_with(tensor_shape.TensorShape([]))

                    # TODO : confirm this noise shape !!
                    noise_shape = array_ops.shape(x)
                    random_tensor = math_ops.divide(1, b)
                    random_tensor += math_ops.divide(b-a, a*b)*random_ops.random_uniform(noise_shape,
                                                                                         seed=self.seed,
                                                                                         dtype=x.dtype)

                    ret = nn.leaky_relu(x, alpha=random_tensor)
                    if not context.executing_eagerly():
                        ret.set_shape(x.get_shape())

                    return ret

        def dropped_inputs_testing():
            with ops.name_scope("random_relu_testing"):
                # in testing phase, deterministic activation function
                # leaky ReLU with slope = 1. - p
                return nn.leaky_relu(inputs, alpha=1/(0.5*(self.a+self.b)))

        with ops.name_scope(self.name, "random-relu", [inputs]):
            output = tf_utils.smart_cond(training,
                                         true_fn=dropped_inputs_training,
                                         false_fn=dropped_inputs_testing)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'a': self.a,
            'b': self.b,
            'seed': self.seed
        }

        base_config = super(RandomizedReLUKeras, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomizedReLUTensorFlow(RandomizedReLUKeras, base.Layer):
    def __init__(self, a=3, b=8, seed=None, name=None, **kwargs):
        super(RandomizedReLUTensorFlow, self).__init__(a=a, b=b, seed=seed, name=name)

    def call(self, inputs, training=False):
        return super(RandomizedReLUTensorFlow, self).call(inputs, training=training)


def randomized_relu(inputs, a=3, b=8, training=False, seed=None, name=None):
    layer = RandomizedReLUTensorFlow(a=a, b=b, seed=seed, name=name)
    return layer.call(inputs, training=training)