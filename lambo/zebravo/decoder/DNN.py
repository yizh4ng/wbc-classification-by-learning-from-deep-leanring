import os, sys
import cupy as np
from tframe import tf
from tframe import pedia
from lambo.zebravo.decoder.decoder import Decoder
from ditto.image.interferogram import Interferogram

import time
from ditto import ImageConfig






class DNN(Decoder):
  def __init__(self):
    super(DNN, self).__init__()

    base_dir = 'E:\projects\lambai'
    sys.path.append(base_dir)
    sys.path.append(base_dir + '/01-PR/')

    from tframe import Predictor, DataSet
    import t1_scaled_unet as task
    from pr_core import th
    th.train = False
    task.main(None)
    xin_net: Predictor = th.model()
    xin_net.launch_model(False)
    self.fetch_list = xin_net.val_outputs.tensor

    default_feed_collection = tf.get_collection(pedia.default_feed_dict)
    self.is_training = tf.get_collection(pedia.is_training)[0]

    self.sess = xin_net.session

    for tensor in default_feed_collection:
      if 'input' in tensor.name.lower():
        self.input_tensor = tensor

    self.img_size = ImageConfig['img_size']
    self.thread_id = 3
    self.low_frequency=None

  # def preprocess(self, image, mean=None):
  #   if self.low_frequency is None:
  #     self.low_frequency = np.array(Interferogram(image, radius=200).low_frequency_filter)
  #   image = np.array(image) - self.low_frequency
  #   # image = np.array(image) - np.zeros_like(image)
  #   # image -= np.min(image)
  #   if mean == None:
  #     mean = np.mean(image)
  #   # image_copy = np.copy(image)
  #   zero_index = image < mean
  #   one_index = image > mean
  #   image[zero_index] = 0
  #   image[one_index] = 1
  #   return image

  def analyze(self,davincv):
    # img_arr = np.array(img)
    # mu = np.mean(img_arr)
    # sigma = np.std(img_arr)
    # img_arr = (img_arr - mu) / sigma
    # img_arr = np.reshape(img, (4, 512, 640, 1))
    # img = self.preprocess(img)
    img = davincv.objects[davincv.raw_channel]
    img_arr = np.reshape(img, (1, *img.shape, 1))
    tic = time.time()
    result = self.sess.run(self.fetch_list, {self.input_tensor: img_arr, self.is_training: False})
    result = np.reshape(-np.array(result), (1, *img.shape, 1)).get()
    # print(time.time() - tic)
    return result[0, :, :, 0]

  # def decode(self, img, bg_ig):
  #   self.frame += 1
  #   return self._decode(img)

# if __name__ == '__main__':
#   import cupy as np
#   print(decode(np.ones((512,512))).get().shape)