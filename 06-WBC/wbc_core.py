import sys, os
import numpy as np
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)

sys.path.append(os.path.join(sys.path[0], 'roma'))
sys.path.append(os.path.join(sys.path[0], 'ditto'))

os.environ["CUDA_VISIBLE_DEVICES"]= "0" if not 'home' in sys.path[0] else "1"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if not 'home' in sys.path[0]:
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 3)]
  )
else:
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]
  )
# =============================================================================
import tframe
tframe.set_random_seed(404)
from tframe import console
from tframe.core.agent import Agent
from tframe.trainers.trainer import Trainer
# from tframe.configs.trainerhub import TrainerHub
from wbc.wbc_config import WBCConfig as TrainerHub
from tframe.data.augment.img_aug import image_augmentation_processor, alter
from tframe.data.prepocess.images import rotagram

import wbc_du as du
# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = TrainerHub()
# -----------------------------------------------------------------------------

# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.raw_data_dir = r'G:\projects\data\wbc'
th.input_shape = [300, 300, 1]
th.non_train_input_shape = th.input_shape

# -----------------------------------------------------------------------------
# set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 30

th.print_cycle = 10
th.validation_per_round = 1
th.probe = True
th.probe_cycle = 1

th.val_size = 0.2
th.val_batch_size = 32
th.eval_batch_size = 32
th.val_progress_bar = True

th.validate_test_set = True
th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True
th.allow_growth = False

th.export_tensors_upon_validation = True
th.allow_activation = True

from tframe.data.dataset import DataSet


from tframe.data.feature_extractor.cell import Variance, RegionSize, \
  RegionIntegrate, RegionVariance, Maximum
FEATURES = [
  Variance(),
  RegionSize(low=4),
  RegionSize(high=0.1),
  # RegionSize(high=0.2), #
  RegionSize(high=3),
  # RegionVariance(low=0.1), #
  RegionVariance(low=0.2),
  # RegionVariance(low=2), #
  # RegionVariance(low=3), #
  # RegionVariance(high=0.1), #
  RegionVariance(high=0.2),
  RegionIntegrate(low=4),
  # RegionIntegrate(low=3), #
  # RegionIntegrate(low=2), #
  # RegionIntegrate(low=0.1), #
  RegionIntegrate(low=0),
  Maximum(),
]
FEATURES = [Variance(), Maximum()]
def preprocess(dataset: DataSet):
  console.show_status('Preprocessing {}...'.format(dataset.name))
  if th.use_meta_data:
    def normalize(x):
      x -= np.mean(x)
      sigma = np.std(x)
      return x / sigma

    features = []
    for feature_extractor in FEATURES:
      feature = feature_extractor(dataset)
      features.append(normalize(feature))
    features = np.array(features)
    dataset.features = np.transpose(features, axes=(1, 0, 2))
    dataset.features = np.sum(dataset.features, axis=-1)

  if th.no_background:
    dataset.features[dataset.features < 0.2] = 0

  if th.no_cell:
    dataset.features[dataset.features > 0.2] = 0

  if th.data_config == 'alpha':
    CLASSES = ['B', 'T', 'Monocyte', 'Granulocyte', 'CD4', 'CD8']
    dataset = dataset.sub_group('cell_class',
                                [CLASSES[0], CLASSES[1], CLASSES[2],
                                 CLASSES[3]])

    def target_mapping(x):
      if x in ('B', 'T'):
        return 'lymphocytes'
      else:
        return x

    dataset.set_classification_target('cell_class',
                                      ['lymphocytes', 'Monocyte', 'Granulocyte'],
                                      func=target_mapping)
    return dataset
  elif th.data_config == 'beta':
    CLASSES = ['B', 'T', 'Monocyte', 'Granulocyte', 'CD4', 'CD8']
    dataset = dataset.sub_group('cell_class', [CLASSES[0], CLASSES[1]])
    def target_mapping(x):
      if x == 'B':
        return 'B cells'
      else:
        return 'T cells'
    dataset.set_classification_target('cell_class', func=target_mapping)
    return dataset
  elif th.data_config == 'gamma':
    CLASSES = ['CD4', 'CD8']
    dataset = dataset.sub_group('cell_class', [CLASSES[0], CLASSES[1]])
    dataset.set_classification_target('cell_class')
    return dataset
  else:
    raise NotImplementedError


def activate():
  if not th.allow_activation:
    return
  # build model
  assert callable(th.model)
  model = th.model()
  mark ='f{}-k{}-lr{}-bs{}-rt{}-aug{}{}-a{}-{}-dc{}-new'.format(
    th.filters, th.kernel_size, th.learning_rate, th.batch_size,
    th.rotagram, th.augmentation,th.aug_config, th.archi_string, th.dense_string,
    th.data_config)
  if th.no_background:
    mark += 'nbk'
  if th.no_cell:
    mark += 'nc'
  agent = Agent(mark, th.task_name)
  agent.config_dir(__file__)
  th.data_dir = agent.data_dir

  # load data
  train_set, test_set = du.load_data(th)
  assert isinstance(train_set, DataSet)

  train_set = preprocess(train_set)
  test_set = preprocess(test_set)
  val_set, train_set = train_set.split([th.val_size],
                                       ['Val set', 'Train set'],
                                       random=True, over_key='cell_class')
  train_set, _ = train_set.split([50], ['train', 'remain'], random=True,
                                 over_key='cell_class')

  if th.augmentation:
    if th.rotagram:
      train_set.append_batch_preprocessor(lambda data_set, is_training:
                                          alter(data_set, is_training, 'lr'))
    else:
      train_set.batch_preprocessor = lambda data_set, is_training:\
        image_augmentation_processor(aug_config=th.aug_config,
                                     data_batch=data_set, is_training=is_training)


  train_set.report()
  # train_set.view()
  val_set.report()
  # val_set.view()
  test_set.report()
  # test_set.view()

  def probe(trainer:Trainer):
    import tensorflow.keras as keras
    assert isinstance(trainer.model.keras_model, keras.Model)
    if not hasattr(trainer, 'print_variable'):
      trainer.print_variable = []
      for var in trainer.model.keras_model.trainable_variables:
        if 'lf_' in var.name:
          trainer.print_variable.append(var)
    to_print = []
    for var in trainer.print_variable:
      to_print.append('{}: {:.3}'.format(var.name.split('/')[1], var.numpy()))
    print(to_print)

  # Train or evaluate
  if th.train:
    trainer = Trainer(model, agent, th, train_set, val_set, test_set)
    trainer.probe = lambda :probe(trainer)
    trainer.train()
    model.keras_model, _ = trainer.agent.load_model('model')
    model.evaluate(train_set, batch_size=th.eval_batch_size)
    model.evaluate(val_set, batch_size=th.eval_batch_size)
    model.evaluate(test_set, batch_size=th.eval_batch_size)
  else:
    trainer = Trainer(model, agent, th, train_set, val_set, test_set)
    model.keras_model, _ = trainer.agent.load_model('model')
    model.keras_model.summary()
    # model.show_feauture_maps(test_set.split([12], names=['small', 'rest'],
    #                                         over_key='cell_class')[0],
    #                          class_key='cell_class')
    # model.evaluate(train_set, batch_size=th.eval_batch_size)
    # model.evaluate(val_set, batch_size=th.eval_batch_size)
    model.evaluate(test_set, batch_size=th.eval_batch_size)

    np.random.seed(4444)
    # test_set = test_set.split([12], names=['small', 'rest'],
    #                           over_key='cell_class')[0]
    # test_set.batch_preprocessor = lambda data_set, is_training: \
    #   image_augmentation_processor(aug_config=th.aug_config,
    #                                data_batch=data_set,
    #                                is_training=is_training)
    # test_set = next(test_set.gen_batches(-1,is_training=True))

    model.show_heatmaps_on_dataset(test_set)
    # model.show_activation_maximum(test_set)
    # model.show_feature_space(test_set, batch_size=th.batch_size)
    # false_set.view()
  # End
  console.end()
