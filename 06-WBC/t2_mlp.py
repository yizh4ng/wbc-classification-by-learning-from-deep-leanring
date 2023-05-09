import wbc_core as core

from wbc_core import th

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.net import Net
from tframe.nets.image2scalar.resnet import ResNet
import tensorflow.keras as keras
from tframe.models.classifier import Classifier
from tframe.quantity import *
from wbc.arch import LFLayer

# import numpy as np
# np.random.seed(0)

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
def model(th):
  net = Net(name=model_name)
  if not th.use_meta_data:
    net.add(LFLayer(th.feature_str))
  if th.dense_string != 'x':
    for neuron_num in th.dense_string.split('-'):
      net.add(keras.layers.Dense(int(neuron_num), use_bias=True))
      net.add(tf.keras.layers.Activation('relu'))

  net.add(keras.layers.Dense(2, use_bias=True))
  net.add(keras.layers.Activation('softmax'))

  model = Classifier(CrossEntropy(), [Accraucy(), MSE(), CrossEntropy()],
                     net, name=model_name)
  return model


def main(_):
  console.start('{} on WBC task'.format(model_name.upper()))
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.rotagram = False
  th.feature_key = 'phase'
  th.data_config = 'beta'
  th.use_meta_data = False
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.input_shape = [len(core.FEATURES)] if th.use_meta_data else [300, 300, 1]
  th.non_train_input_shape = th.input_shape
  th.model = lambda : model(th)
  th.dense_string = '8-32-32-8'
  # th.feature_str = 'M-V-V>-V<-I>-I<-P>-P<'
  th.feature_str = 'M-V-P>-P>-P<-P<-V>-V>-V<-V<-I>-I>-I<-I<'
  # th.feature_str = 'M-V'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100000000000000
  th.batch_size = 32
  th.learning_rate = 0.0003
  # th.learning_rate = 0.3

  th.patience = 10000
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.rehearse = False
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.augmentation = False
  th.aug_config = None

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  main(None)
