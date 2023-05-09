import tensorflow as tf
import os, sys
import re
from tframe.utils import Note
from tframe import console
from tframe.utils.local import check_path, clear_paths
from tframe.utils.file_tools import io_utils
from tensorflow import keras
import shutil



class Agent(object):
  def __init__(self, mark, task_name, max_num_saved_models=2):
    self.saved_model_paths = []
    self.mark = mark
    self.task_name = task_name
    # self.config_dir()
    self.summary_writer = None
    self.note = Note()
    self.max_num_saved_models = max_num_saved_models

  @property
  def root_path(self):
   return self.job_dir

  @property
  def data_dir(self):
    return check_path(self._data_dir)

  @data_dir.setter
  def data_dir(self, data_dir):
    self._data_dir = check_path(data_dir)

  @property
  def log_dir(self):
    return check_path(self.root_path, 'logs',
                      self.mark)

  @property
  def ckpt_dir(self):
    return check_path(self.root_path, 'checkpoints',
                      self.mark)


  @property
  def snapshot_dir(self):
    return check_path(self.root_path, 'snapshot',
                      self.mark)

  @property
  def gather_summ_path(self):
    return os.path.join(check_path(self.root_path), 'gather.sum')

  def clear_dirs(self):
    paths = [self.snapshot_dir, self.ckpt_dir, self.log_dir]
    for path in paths:
      # clear_paths(path)
      shutil.rmtree(path)


  def config_dir(self, __file__, dir_depth=2):
    """This method should be called only in XX_core.py module for setting
       default job_dir and data_dir.
    """
    ROOT = os.path.abspath(__file__)
    for _ in range(dir_depth):
      ROOT = os.path.dirname(ROOT)
      if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
    self.job_dir = os.path.join(sys.path[dir_depth - 1])
    self._data_dir = os.path.join(self.job_dir, 'data')
    self.job_dir = os.path.join(self.job_dir, self.task_name)
    console.show_status('Job directory set to `{}`'.format(self.job_dir))


  def save_model(self, model, counter, mark=None,
                       suffix='.sav'):
    if len(self.saved_model_paths) >= self.max_num_saved_models:
      saved_model_to_delete = self.saved_model_paths.pop(0)
      shutil.rmtree(saved_model_to_delete)
    file_name = 'model-c{}{}'.format(counter, suffix)
    if mark is not None:
      file_name += '-{}'.format(mark)

    path =check_path(self.ckpt_dir, file_name)

    model.save(path)
    self.saved_model_paths.append(path)

  def save_figure(self, figure, file_name, clim=None, cb=True, axis_off=False):
    import matplotlib.pyplot as plt
    save_path = os.path.join(self.snapshot_dir, file_name + '.png')
    if not os.path.exists(save_path):
      plt.imshow(figure)
      if clim is not None:
        plt.clim(clim[0], clim[1])
      if cb:
        plt.colorbar()
      if axis_off:
        plt.axis('off')
      plt.savefig(save_path, bbox_inches='tight',transparent=True, pad_inches=0)
      plt.close()

  def load_model(self, mark=None, suffix='.sav'):
    counter = 0
    for root, dirs, files in os.walk(self.ckpt_dir):
      for dir in dirs:
        if mark is not None and mark not in dir: continue
        _counter = self.get_model_counter_from_name(dir)
        if _counter is not None:
          if _counter > counter:
            counter = _counter
    file_name = 'model-c{}{}'.format(counter, suffix)
    if mark is not None:
      file_name += '-{}'.format(mark)
    path = check_path(self.ckpt_dir, file_name)
    return keras.models.load_model(path), counter


  def get_model_counter_from_name(self, path):
    matched = re.search('-c(\d+)', path)
    if matched:
      counter = matched.group(1)
      return int(counter)
    else:
      return None

  def write_model_summary(self):
    # TODO: Use summary, now use plot_model
    if self.summary_writer is None:
      self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    with self.summary_writer.as_default():
      tf.summary.trace_export(name='model_structure', step=0,
                              profiler_outdir=self.log_dir)

  def write_summary(self, name:str, value, step):
    if self.summary_writer is None:
      self.summary_writer = tf.summary.create_file_writer(self.log_dir)
    with self.summary_writer.as_default():
      tf.summary.scalar(name, value, step)
      self.summary_writer.flush()

  def write_summary_from_dict(self, dict:dict, step:int,
                              name_scope:str = ''):
    suffix = ''
    if name_scope != '':
      suffix = '{}/'.format(name_scope)

    for key in dict:
      self.write_summary(suffix+key.name, dict[key], step)

  def gather_to_summary(self):
    import pickle
    # Try to load note list into summaries
    file_path = self.gather_summ_path
    if os.path.exists(file_path):
      # with open(file_path, 'rb') as f: summary = pickle.load(f)
      summary = io_utils.load(file_path)
      assert len(summary) > 0
    else: summary = []
    # Add note to list and save
    note = self.note
    summary.append(note)
    io_utils.save(summary, file_path)

    # Show status
    console.show_status('Note added to summaries ({} => {}) at `{}`'.format(
      len(summary) - 1, len(summary), file_path))

  def create_bash(self):
    command = 'tensorboard --logdir=./logs/ --port={}'.format(6006)
    file_path = check_path(self.root_path, create_path=True)
    file_names = ['win_launch_tensorboard.bat', 'unix_launch_tensorboard.sh']
    for file_name in file_names:
      path = os.path.join(file_path, file_name)
      if not os.path.exists(path):
        f = open(path, 'w')
        f.write(command)
        f.close()
