from collections import OrderedDict
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam


from tframe import console, DataSet
from tframe.quantity import Quantity
from tframe.trainers.trainer import Trainer



class GANTrainer(Trainer):
  def __init__(
      self,
      generator,
      discriminator,
      agent,
      config,
      training_set=None,
      valiation_set=None,
      test_set=None,
      probe=None
  ):
    super(GANTrainer, self).__init__(None, agent, config, training_set,
                                     valiation_set, test_set, probe)
    self.generator = generator
    self.discriminator = discriminator
    self.dics_opt = Adam(self.th.learning_rate)
    self.gen_opt_1 = Adam(self.th.learning_rate)
    self.gen_opt_2 = Adam(self.th.learning_rate)
    self.model = self.generator

  def train(self):
    # :: Before training
    if self.th.overwrite:
      self.agent.clear_dirs()

    if self.th.load_model:
      self.generator.keras_model, self.counter = self.agent.load_model(
        'gen')
      self.discriminator.keras_model, self.counter = self.agent.load_model(
        'dis')
      console.show_status(
        'Model loaded from counter {}'.format(self.counter))
    else:
      self.generator.build(self.th.input_shape)
      self.discriminator.build((self.th.win_size,self.th.win_size, 2))
      tf.summary.trace_on(graph=True, profiler=True)

      # self.model.link(tf.random.uniform((self.th.batch_size, *self.th.input_shape)))
      # _ = self.model.keras_model(tf.keras.layers.Input(self.th.input_shape))
      @tf.function
      def predict(x):
        self.generator.keras_model(x)
        self.discriminator.keras_model(tf.random.uniform((self.th.batch_size,
                                                         self.th.win_size,self.th.win_size,2)))
        return None

      predict(tf.random.uniform(
        (self.th.batch_size, *self.th.non_train_input_shape)))
      self.agent.write_model_summary()
      tf.summary.trace_off()
      console.show_status('Model built.')
    self.generator.keras_model.summary()
    self.discriminator.keras_model.summary()
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
      self.agent.save_model(self.discriminator.keras_model,
                            self.counter, 'dis_last')
      self.agent.save_model(self.generator.keras_model,
                            self.counter, 'gen_last')

    self.generator.keras_model, _ = self.agent.load_model('gen')
    self.discriminator.keras_model, _ = self.agent.load_model('dis')
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

    console.show_status('Training ends at round {}'.format(rnd),
                        symbol='[Patience]')

    if self.th.gather_note:
      self.agent.note.put_down_criterion('Total Parameters of Generator',
                                         self.generator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Parameters of Discriminator',
                                         self.discriminator.num_of_parameters)
      self.agent.note.put_down_criterion('Total Iterations', self.counter)
      self.agent.note.put_down_criterion('Total Rounds', rnd)
      # Evaluate the best model if necessary
      ds_dict = OrderedDict()
      ds_dict['Train'] = self.training_set
      ds_dict['Val'] = self.validation_set
      ds_dict['Test'] = self.test_set
      if len(ds_dict) > 0:
        # Load the best model
        if self.th.save_model:
          self.generator.keras_model, self.counter = self.agent.load_model(
            'gen')
          self.discriminator.keras_model, self.counter = self.agent.load_model(
            'dis')

        # Evaluate the specified data sets
        for name, data_set in ds_dict.items():
          loss_dict = self.validate_model(data_set,
                                          batch_size=self.th.val_batch_size)
          for key in loss_dict:
            title = '{} {}'.format(name, key.name)
            # print(title, loss_dict[key])
            self.agent.note.put_down_criterion(title, loss_dict[key].numpy())

    return rnd

  # region : Private Methods

  def gp(self, real_img, fake_img, feature):
    e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
    noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
    noise_img = tf.concat((noise_img, feature), axis=-1)
    with tf.GradientTape() as tape:
      tape.watch(noise_img)
      o = self.discriminator.keras_model(noise_img)
    g = tape.gradient(o, noise_img)  # image gradients
    g_norm2 = tf.sqrt(
      tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
    gp = tf.square(g_norm2 - 1.)
    return tf.reduce_mean(gp)

  def _update_model_by_batch(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape(persistent=True) as tape:
      generator_prediction = self.generator(feature)
      generator_loss = self.generator.loss[0](generator_prediction, target)
      loss_dict[self.generator.loss[0]] = generator_loss

      fake_prediction = self.discriminator(tf.concat([generator_prediction, feature], axis=-1))
      real_prediction = self.discriminator(tf.concat([target, feature], axis=-1))

      gen_loss = 0.1 * self.generator.loss[1](fake_prediction)
      loss_dict[Quantity('gen_loss', True)] = tf.reduce_mean(gen_loss)

      intermediate_layer_model = keras.Model(inputs=self.generator.keras_model.input,
                                       outputs=self.generator.keras_model.get_layer('output').output)
      self.intermediate_model_weights = intermediate_layer_model.trainable_variables

      self.esrgan_weights = []
      assert isinstance(self.generator.keras_model, keras.Model)
      flag = False
      for layer in self.generator.keras_model.layers:
        if flag:
          self.esrgan_weights.extend(layer.trainable_variables)
        if layer.name == 'output':
          flag = True

      intermediate_loss = self.generator.loss[0](intermediate_layer_model(feature),
                                                 tf.image.resize(target,(128, 128)))

      # if self.counter < 900:
      #   generator_loss = generator_loss + intermediate_loss
      # else:
      generator_loss = gen_loss + generator_loss#+ intermediate_loss + generator_loss
      discriminator_loss = self.discriminator.loss[0](real_prediction) \
                           + self.discriminator.loss[1](fake_prediction)\
                          # + 10 * self.gp(target, generator_prediction)


      loss_dict[Quantity('dis_loss', True)] = tf.reduce_mean(discriminator_loss)
      loss_dict[Quantity('real_prediction', True)] = tf.reduce_mean(real_prediction, axis=0)
      loss_dict[Quantity('fake_prediction', True)] = tf.reduce_mean(fake_prediction, axis=0)
      loss_dict[Quantity('intermediate_loss', True)] = intermediate_loss

      for metric in self.generator.metrics:
        loss_dict[metric] = metric(generator_prediction, target)

    inte_grads = tape.gradient(intermediate_loss, self.intermediate_model_weights)
    gen_grads = tape.gradient(generator_loss, self.esrgan_weights)
    dis_grads = tape.gradient(discriminator_loss, self.discriminator.keras_model.trainable_variables)

    self.optimizer.apply_gradients(zip(inte_grads, self.intermediate_model_weights))
    self.optimizer.apply_gradients(zip(gen_grads, self.esrgan_weights))
    self.optimizer.apply_gradients(zip(dis_grads, self.discriminator.keras_model.trainable_variables))

    return loss_dict


  def validate_model(self, data_set:DataSet, batch_size=None, update_record=True):
    _loss_dict = {}
    for metric in self.generator.metrics:
      _loss_dict[metric] = 0
    batch_size_sum = 0
    for i, data_batch in enumerate(data_set.gen_batches(batch_size,
                                                        is_training=False)):
      target = data_batch.targets
      feature = data_batch.features
      generator_prediction = self.generator(feature)
      for metric in self.generator.metrics:
        _loss_dict[metric] += tf.reduce_mean(metric(generator_prediction, target)) \
                              * data_batch.size
      batch_size_sum += data_batch.size

    for metric in self.generator.metrics:
      _loss_dict[metric] /= batch_size_sum

    if update_record:
      for metric in self.generator.metrics:
        metric.try_set_record(_loss_dict[metric], data_set)
    return _loss_dict

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

    # normal save for n epochs
    if self.th.save_model_cycle > 0 and self.th.save_model:
      if self.counter % self.th.save_model_cycle == 0:
        console.show_status('Saving the model to {}'.format(
          self.agent.ckpt_dir), symbol='[Saving]')
        self.agent.save_model(self.generator.keras_model,
                              self.counter, 'gen')
        self.agent.save_model(self.discriminator.keras_model,
                              self.counter, 'dis')
    # save in early stop setting
    else:
      if self.generator.metrics[0]._record_appears[self.validation_set]:
        self.patenice = self.th.patience
        console.show_status('Record appears', symbol='[Patience]')
        if self.th.save_model:
          console.show_status('Saving the model to {}'.format(
            self.agent.ckpt_dir ), symbol='[Saving]')
          self.agent.save_model(self.generator.keras_model,
                                self.counter, 'gen')
          self.agent.save_model(self.discriminator.keras_model,
                                self.counter, 'dis')
      else:
        self.patenice -= 1
        if self.patenice < 0:
          self.th._stop = True
        else:
          console.show_status('Record does not appear [{}/{}]'.format(
            self.patenice + 1, self.th.patience), symbol='[Patience]')

  def _update_model_by_dataset(self, data_set, rnd):
    for i, batch in enumerate(data_set.gen_batches(
        self.th.batch_size, updates_per_round=self.th.updates_per_round,
        shuffle=self.th.shuffle, is_training=True)):
      self.cursor += 1
      self.counter += 1

      # Update model
      if self.cursor % self.th.discriminator_loop != 0:
        loss_dict = self.update_discriminator_by_data_batch(batch)
        if np.mod(self.counter - 1, self.th.print_cycle) == 0:
          self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)
      else:
        loss_dict = self.update_generator_by_data_batch(batch)
        if np.mod(self.counter - 1, self.th.print_cycle) == 0:
          self._print_progress(i, data_set._dynamic_round_len, rnd, loss_dict)

  def update_discriminator_by_data_batch(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape(persistent=True) as tape:
      generator_prediction = self.generator(feature)


      intermediate_layer_model = keras.Model(inputs=self.generator.keras_model.input,
                                             outputs=self.generator.keras_model.get_layer('output').output)
      intermediate_result = intermediate_layer_model(feature)
      self.intermediate_model_weights = intermediate_layer_model.trainable_variables

      blurred_target = tf.image.resize(intermediate_result, (generator_prediction.shape[1], generator_prediction.shape[2])) # 128, 128
      fake_prediction = self.discriminator(tf.concat([generator_prediction, blurred_target], axis=-1))
      real_prediction = self.discriminator(tf.concat([target, blurred_target], axis=-1))
      self.esrgan_weights = []
      assert isinstance(self.generator.keras_model, keras.Model)
      flag = False
      for layer in self.generator.keras_model.layers:
        if flag:
          self.esrgan_weights.extend(layer.trainable_variables)
        if layer.name == 'output':
          flag = True
      discriminator_loss = self.discriminator.loss[0](real_prediction) \
                           + self.discriminator.loss[1](fake_prediction) \
        # + 10 * self.gp(target, generator_prediction, blurred_target)

      intermediate_loss = self.generator.loss[0](intermediate_result,
                                                 tf.image.resize(target,(intermediate_result.shape[1], intermediate_result.shape[2]))) # 32, 32

      loss_dict[Quantity('dis_loss', True)] = tf.reduce_mean(discriminator_loss)
      loss_dict[Quantity('inter_loss', True)] = tf.reduce_mean(intermediate_loss)
      loss_dict[Quantity('real_prediction', True)] = tf.reduce_mean(real_prediction, axis=0)
      loss_dict[Quantity('fake_prediction', True)] = tf.reduce_mean(fake_prediction, axis=0)

    dis_grads = tape.gradient(discriminator_loss, self.discriminator.keras_model.trainable_variables)
    inte_grads = tape.gradient(intermediate_loss, self.intermediate_model_weights)
    self.dics_opt.apply_gradients(zip(dis_grads, self.discriminator.keras_model.trainable_variables))
    self.gen_opt_1.apply_gradients(zip(inte_grads, self.intermediate_model_weights))


    return loss_dict

  def update_generator_by_data_batch(self, data_batch):
    target = data_batch.targets
    feature = data_batch.features
    loss_dict = {}
    with tf.GradientTape(persistent=True) as tape:
      generator_prediction = self.generator(feature)
      generator_loss = self.generator.loss[0](generator_prediction, target)
      loss_dict[self.generator.loss[0]] = generator_loss

      intermediate_layer_model = keras.Model(inputs=self.generator.keras_model.input,
                                             outputs=self.generator.keras_model.get_layer('output').output)
      intermediate_result = intermediate_layer_model(feature)
      self.intermediate_model_weights = intermediate_layer_model.trainable_variables

      blurred_target = tf.image.resize(intermediate_result, (generator_prediction.shape[1], generator_prediction.shape[2])) # 128
      fake_prediction = self.discriminator(tf.concat([generator_prediction, blurred_target], axis=-1))
      real_prediction = self.discriminator(tf.concat([target, blurred_target], axis=-1))

      gen_loss = 0.001 * self.generator.loss[1](fake_prediction)
      loss_dict[Quantity('gen_loss', True)] = tf.reduce_mean(gen_loss)


      self.esrgan_weights = []
      assert isinstance(self.generator.keras_model, keras.Model)
      flag = False
      for layer in self.generator.keras_model.layers:
        if flag:
          self.esrgan_weights.extend(layer.trainable_variables)
        if layer.name == 'output':
          flag = True

      intermediate_loss = self.generator.loss[0](intermediate_result,
                                                 tf.image.resize(target,(intermediate_result.shape[1], intermediate_result.shape[2]))) # 32
      generator_loss = gen_loss + generator_loss#+ intermediate_loss + generator_loss


      loss_dict[Quantity('real_prediction', True)] = tf.reduce_mean(real_prediction, axis=0)
      loss_dict[Quantity('fake_prediction', True)] = tf.reduce_mean(fake_prediction, axis=0)
      loss_dict[Quantity('intermediate_loss', True)] = intermediate_loss

      for metric in self.generator.metrics:
        loss_dict[metric] = metric(generator_prediction, target)

    inte_grads = tape.gradient(intermediate_loss, self.intermediate_model_weights)
    gen_grads = tape.gradient(generator_loss, self.esrgan_weights)

    self.gen_opt_1.apply_gradients(zip(inte_grads, self.intermediate_model_weights))
    self.gen_opt_2.apply_gradients(zip(gen_grads, self.esrgan_weights))

    return loss_dict
