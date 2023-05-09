from collections import OrderedDict

from tframe import console
import tensorflow as tf
from tframe.data.dataset import DataSet
from tensorflow.keras.optimizers import Adam, SGD
from tframe.core.agent import Agent
import numpy as np

# Only trainer knows the trainer hub right?

class Trainer():
  """Base class of trainer for training tframe models.

     Model save mechanism when save_mode is
       (1) SaveMode.NAIVE:
           Model will be saved only at the end of each round naively
       (2) SaveMode.ON_RECORD:
           Model will be saved only when a new metric record appears
           after model finishes its warm-up rounds
   """
  HubClass = None
  def __init__(
      self,
      model,
      agent,
      config,
      training_set=None,
      validation_set=None,
      test_set=None,
      probe=None
  ):
    # Set model for trainer
    self.model = model

    # Date set attributes
    self._training_set = None
    self._validation_set = None
    self._test_set = None
    self.set_data(training_set, validation_set, test_set)
    self.round = 0
    self.th = config
    self.counter = 0
    self.cursor = None
    self.agent = agent
    self.probe = probe
    self.optimizer = Adam(self.th.learning_rate)
    # self.optimizer = SGD(self.th.learning_rate)

    # Set callable attributes


  # region : Properties

  @property
  def key_metric(self):
    return self.metrics_manager.early_stop_slot

  @property
  def training_set(self):
    if self._training_set is not None:
      assert isinstance(self._training_set, DataSet)
    return self._training_set

  @property
  def validation_set(self):
    if self._validation_set is not None:
      assert isinstance(self._validation_set, DataSet)
    return self._validation_set

  @property
  def test_set(self):
    if self._test_set is not None:
      assert isinstance(self._test_set, DataSet)
    return self._test_set

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None, test_set=None):
    if training_set is not None:
      self._training_set = training_set
    if validation_set is not None:
      self._validation_set = validation_set
    if test_set is not None:
      self._test_set = test_set

  def _inter_cut(self, i, total, content, prompt='>>', start_time=None):
    # Show content
    console.show_status(content, symbol=prompt)
    # Print progress bar
    console.print_progress(i, total, start_time=start_time)
    # self.recover_prototal_roundsgress(start_time)

  def recover_progress(self, start_time=None):
    # Print progress bar
    if self.th.progress_bar and self.th.round_length is not None:
      assert isinstance(self._training_set, DataSet)
      progress = self.th.round_progress
      assert progress is not None
      console.print_progress(progress=progress, start_time=start_time)

  @staticmethod
  def _dict_to_string(dict_, data_set):
    assert isinstance(dict_, dict)
    string_array = []
    for k, v in dict_.items():
      # _v = tf.reduce_mean(v)
      # _v = v
      try:
        _v = '{:.3f}'.format(v)
      except ValueError:
        _v = v
      # string = '{} = {:.3f}'.format(k.name, _v)
      string = '{} = {}'.format(k.name, _v)
      if k._record_appears[data_set]:
        string +=' [New Record]'
      string_array.append(string)
    return ', '.join(string_array)

  def _print_progress(self, i, total, rnd, loss_dict):
    content = '{} {} '.format(
      self.th.round_name, rnd, loss_dict)
    content += self._dict_to_string(loss_dict, self.training_set)

    # Show time elapsed for a single update if required

    self._inter_cut(i, total, content, prompt='[Train]', start_time=self.th.start_time)
  # endregion : Public Methods

  # region : Train

  def train(self):
    # :: Before training
    if self.th.overwrite:
      self.agent.clear_dirs()

    if self.th.load_model:
      # self.model.build(self.th.input_shape)
      self.model.keras_model, self.counter = self.agent.load_model(self.model.mark)
      console.show_status('Model loaded from counter {}'.format(self.counter))
    else:
      self.model.build(self.th.input_shape)
      tf.summary.trace_on(graph=True, profiler=True)
      # self.model.link(tf.random.uniform((self.th.batch_size, *self.th.input_shape)))
      # _ = self.model.keras_model(tf.keras.layers.Input(self.th.input_shape))
      @tf.function
      def predict(x):
        return self.model.keras_model(x)
      predict(tf.random.uniform((self.th.batch_size, *self.th.non_train_input_shape)))
      self.agent.write_model_summary()
      tf.summary.trace_off()
      console.show_status('Model built.')
    self.model.keras_model.summary()
    self.agent.create_bash()

    # :: During training
    if self.th.rehearse:
      return



    rounds = self._outer_loop()

    # :: After training
    # self._end_training(rounds)
    # Put down key configurations to note
    self.agent.note.put_down_configs(self.th.key_options)
    # Export notes if necessary
    # Gather notes if necessary
    if self.th.gather_note:
      self.agent.gather_to_summary()

    if self.th.save_last_model:
      self.agent.save_model(self.model.keras_model,
                            self.counter, 'model')

    self.model.keras_model, _ = self.agent.load_model('model')
    if self.th.probe:
      self.probe()

  # region : During training

  def _outer_loop(self):
    rnd = 0
    self.patenice = self.th.patience
    for _ in range(self.th.epoch):
      rnd += 1
      console.section('round {}:'.format(rnd))

      self._inner_loop(rnd)
      self.round += 1
      if self.th.probe and rnd % self.th.probe_cycle == 0:
        assert callable(self.probe)
        self.probe()

      if self.th._stop:
        break

    console.show_status('Training ends at round {}'.format(rnd), symbol='[Patience]')



    if self.th.gather_note:
      self.agent.note.put_down_criterion('Total Parameters', self.model.num_of_parameters)
      self.agent.note.put_down_criterion('Total Iterations', self.counter)
      self.agent.note.put_down_criterion('Total Rounds', rnd)
      # Evaluate the best model if necessary
      ds_dict = OrderedDict()
      # ds_dict['Train'] = self.training_set
      ds_dict['Val'] = self.validation_set
      ds_dict['Test'] = self.test_set
      if len(ds_dict) > 0:
        # Load the best model
        if self.th.save_model:
          self.model.keras_model, self.counter = self.agent.load_model(
            'model')
        # Evaluate the specified data sets
        for name, data_set in ds_dict.items():
          loss_dict = self.validate_model(data_set,
                                          batch_size=self.th.val_batch_size)
          for key in loss_dict:
            title = '{} {}'.format(name, key.name)
            # print(title, loss_dict[key])
            self.agent.note.put_down_criterion(title, loss_dict[key].numpy())

    return rnd

  def _inner_loop(self, rnd):
    self.cursor = 0
    self._record_count = 0
    self._update_model_by_dataset(self.training_set, rnd)

    if self.th.validate_train_set:
     loss_dict = self.validate_model(self.training_set,
                                     batch_size=self.th.val_batch_size, update_record=False)
     self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='train')

     console.show_status('Train set: ' +self._dict_to_string(loss_dict, self.training_set), symbol='[Validation]')

    if self.th.validate_val_set:
      loss_dict = self.validate_model(self.validation_set,
                                      batch_size=self.th.val_batch_size)
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='validation')

      console.show_status('Validation set: ' + self._dict_to_string(loss_dict,
                                                                    self.validation_set),
                          symbol='[Validation]')

    if self.th.validate_test_set:
      loss_dict = self.validate_model(self.test_set,
                                      batch_size=self.th.val_batch_size)
      console.show_status('Test set: ' + self._dict_to_string(loss_dict,
                                                              self.test_set),
                          symbol='[Validation]')
      self.agent.write_summary_from_dict(loss_dict, rnd, name_scope='test')

    # self.th._stop = True #Test

    if self.model.metrics[0]._record_appears[self.validation_set]:
      self.patenice = self.th.patience
      console.show_status('Record appears', symbol='[Patience]')
      if self.th.save_model:
        console.show_status('Saving the model to {}'.format(
          self.agent.ckpt_dir ), symbol='[Saving]')
        self.agent.save_model(self.model.keras_model,
                                      self.counter, 'model')
    else:
      self.patenice -= 1
      if self.patenice < 0:
        self.th._stop = True
      else:
        console.show_status('Record does not appear [{}/{}]'.format(
          self.patenice + 1, self.th.patience), symbol='[Patience]')

  # endregion : During training

  # region : After training

  # endregion : After training

  # endregion : Train

  # region : Private Methods

  def _update_model_by_batch(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape() as tape:
      prediction = self.model(feature)
      loss = self.model.loss(prediction, target)
      loss_dict[self.model.loss] = loss
      for metric in self.model.metrics:
        loss_dict[metric] = metric(prediction, target)
    grads = tape.gradient(loss, self.model.keras_model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.keras_model.trainable_variables))
    return loss_dict

  def _update_model_by_dataset(self, data_set, rnd):
    for i, batch in enumerate(data_set.gen_batches(
        self.th.batch_size, updates_per_round =self.th.updates_per_round,
        shuffle=self.th.shuffle, is_training=True)):

      self.cursor += 1
      self.counter += 1
      # Update model
      loss_dict = self._update_model_by_batch(batch)
      if np.mod(self.counter - 1, self.th.print_cycle) == 0:
        self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)

  def validate_model(self, data_set:DataSet, batch_size=None, update_record=True):
    _loss_dict = {}
    _loss_dict[self.model.loss] = 0
    for metric in self.model.metrics:
      metric_key = metric
      _loss_dict[metric_key] = 0

    batch_size_sum = 0
    for i, data_batch in enumerate(data_set.gen_batches(batch_size,
                                                        is_training=False)):
      target = data_batch.targets
      feature = data_batch.features
      prediction = self.model(feature)
      loss = self.model.loss(prediction, target)

      _loss_dict[self.model.loss] += tf.reduce_mean(loss) * data_batch.size
      for metric in self.model.metrics:
        _loss_dict[metric] += tf.reduce_mean(metric(prediction, target))\
                                 * data_batch.size
      batch_size_sum += data_batch.size

    _loss_dict[self.model.loss] /= batch_size_sum
    for metric in self.model.metrics:
      _loss_dict[metric] /= batch_size_sum

    if update_record:
      self.model.loss.try_set_record(_loss_dict[self.model.loss], data_set)
      for metric in self.model.metrics:
        metric.try_set_record(_loss_dict[metric], data_set)
    return _loss_dict




