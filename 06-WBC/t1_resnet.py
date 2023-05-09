import wbc_core as core

from wbc_core import th

from tframe import tf

from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.net import Net
from tframe.nets.image2scalar.resnet import ResNet
from tensorflow import keras
from tframe.models.classifier import Classifier
from tframe.quantity import *

# import numpy as np
# np.random.seed(0)

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
th.task_name = 'resnet'
def model(th):
  net = Net(name='resnet')
  # net.add(keras.layers.Conv2D(th.filters, 7, 2, padding='same'))
  # net.add(keras.layers.MaxPooling2D())
  net.add(ResNet(th.filters, th.kernel_size, th.archi_string, use_bias=False, padding='same'))

  if th.dense_string != 'x':
    for neuron_num in th.dense_string.split('-'):
      net.add(keras.layers.Dense(int(neuron_num), use_bias=True))

  net.add(keras.layers.Dense(2, use_bias=True))

  model = Classifier(CrossEntropy(), [Accraucy(), MSE(), CrossEntropy()],
                     net, name='resnet')
  return model


def main(_):
  console.start('{} on WBC task'.format('resnet'.upper()))
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.rotagram = False
  th.feature_key = 'phase'
  th.data_config = 'beta'
  th.no_background = False
  th.no_cell = False
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = lambda : model(th)
  th.kernel_size = 3
  th.filters = 64
  # th.filters = 1
  th.archi_string = '1-2-1'
  # th.archi_string = '1-1'
  th.dense_string = 'x'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32
  th.learning_rate = 0.0003
  # th.shuffle = True
  # th.updates_per_round = 60

  th.patience = 30
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
  th.augmentation = True
  th.aug_config = 'flip-rotate'

  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  main(None)
