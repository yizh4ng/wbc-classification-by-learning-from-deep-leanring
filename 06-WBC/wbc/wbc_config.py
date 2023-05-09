from tframe.configs.config_base import Flag
from tframe.configs.trainerhub import TrainerHub



class WBCConfig(TrainerHub):
  # num_classes = Flag.integer(2, 'Number of classes', is_key=None)
  target_key = Flag.string('cell_class', 'The target key to set label', is_key=None)
  dense_string = Flag.string('x', 'Architecture for the dense layers', is_key=None)
  rotagram = Flag.boolean(False, 'Whether to use rotagrams', is_key=None)
  feature_key = Flag.string('phase', 'The feature key to set features', is_key=None)
  data_config = Flag.string('alpha', 'Data config', is_key=None)
  use_meta_data = Flag.boolean(False, 'Whether to use rotagrams', is_key=None)
  no_background = Flag.boolean(False, 'Whether cancel background', is_key=None)
  no_cell = Flag.boolean(False, 'Whether cancel cell region', is_key=None)

WBCConfig.register()
