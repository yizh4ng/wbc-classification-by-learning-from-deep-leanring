import tensorflow as tf
import tensorflow.keras as keras
from tframe.core import Function

def region_larger_than(features, x, alpha=0.1):
  return tf.sigmoid(2 * (tf.maximum((features-x) * alpha, (features-x)) )) #+ x

def region_smaller_than(features, x, alpha=0.1):
  return tf.sigmoid(2 * (-tf.minimum((features-x) * alpha, (features-x)) )) #+ x

def variance(features, mask=None):
  # return tf.linalg.normalize(tf.math.reduce_variance(features, axis=(1,2)), axis=0)[0] + 1
  # return tf.math.reduce_mean(tf.square(features - tf.reduce_mean(features, axis=(1,2),keepdims=True)) * features, axis=(1,2))/tf.math.reduce_mean(features, axis=(1,2))
  if mask is None:
    return tf.math.reduce_variance(features, axis=(1, 2))
  else:
    return tf.math.reduce_mean(tf.square(features - tf.reduce_mean(features, axis=(1,2),keepdims=True)) * mask, axis=(1,2))/tf.math.reduce_mean(mask, axis=(1,2))


def maximum(features):
  max = tf.math.reduce_max(features, axis=(1,2))
  return max
  # return tf.linalg.normalize(tf.math.reduce_max(features, axis=(1, 2)), axis=0)[0] + 1

def integrate(features, mask):
  return tf.math.reduce_sum(features * mask, axis=(1,2))/tf.math.reduce_sum(mask, axis=(1,2))
  # return tf.linalg.normalize(tf.math.reduce_sum(features, axis=(1,2)), axis=0)[0] + 1
  # return tf.math.reduce_sum(features, axis=(1,2))
# tf.reshape(features, (features.shape[0],
    #                       features.shape[1] * features.shape[2])), axis=-1)
def proportion(features, mask):
  return tf.math.reduce_sum(mask, axis=(1,2))/ 90000
  # return tf.linalg.normalize(tf.math.reduce_sum(features, axis=(1,2))/ (tf.math.reduce_mean(features, axis=(1,2)) * 300 * 300))[0] + 1

class LFLayer(keras.layers.Layer):
# class LFLayer(Function):

  def __init__(self, feature_str:str):
    super(LFLayer, self).__init__()
    self.feature_str = feature_str


  def build(self, input_shape):
    self.new_weights = []
    weight_index = 0
    for _str in self.feature_str.split('-'):
      if '<' in _str:
        self.new_weights.append(self.add_weight(
          name='lf_{}'.format(weight_index), trainable=True, initializer=keras.initializers.constant(2),
          # initializer=tf.keras.initializers.RandomNormal(mean=2, stddev=1, seed=None),
          constraint=lambda x: tf.clip_by_value(x, 0.2, 4)))
        weight_index += 1
      elif '>' in _str:
        self.new_weights.append(self.add_weight(
          name='lf_{}'.format(weight_index),trainable=True,initializer=keras.initializers.constant(2),
          # initializer=tf.keras.initializers.RandomNormal(mean=2, stddev=1, seed=None),
          constraint=lambda x: tf.clip_by_value(x, 0.2, 4)))
        weight_index += 1
    super(LFLayer, self).build(input_shape)


  @property
  def feature_extractor_dict(self):
    return {'M': maximum, 'V': variance, 'I': integrate, 'P':proportion}

  def call(self, *inputs):
    # region : Check inputs
    if len(inputs) == 1:
      input = inputs[0]
    else:
      raise SyntaxError('!! Too much inputs')
    features = []
    weight_index = 0
    for str in self.feature_str.split('-'):
      if str == 'M':
        features.append(maximum(input))
      elif str == 'V':
        features.append(variance(input))
      elif '<' in str:
        mask = region_smaller_than(input, self.new_weights[weight_index])
        features.append(self.feature_extractor_dict[str.split('<')[0]](input, mask))
        weight_index+=1
      elif '>' in str:
        mask = region_larger_than(input, self.new_weights[weight_index])
        features.append(self.feature_extractor_dict[str.split('>')[0]](input, mask))
        weight_index+=1
      else:
        raise SyntaxError('? {}'.format(self.feature_str))
    features = tf.concat(features, axis=-1)
    return features
