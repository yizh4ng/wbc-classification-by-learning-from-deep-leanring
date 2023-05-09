import sys, os
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)

print(sys.path)
from wbc_core import th, du
from tframe.data.feature_extractor.cell import *
import tensorflow.keras as keras
import tensorflow as tf


th.data_config = 'alpha'
# th.data_dir = '/home/yiz/share/projects/lambai_v2/06-WBC/data'
# model_path =r'/home/yiz/share/projects/lambai_v2/06-WBC/resnet/checkpoints/f1-k3-lr0.0003-bs32-rtFalse-augTrue-a1-1-x-dcalpha/model-c4320.sav'
th.data_dir = '../data'
# model_path = '../resnet/checkpoints/f1-k3-lr0.0003-bs32-rtFalse-augTrueflip-rotate-a1-1-x-dcalpha/model-c4320.sav'
model_path = '../resnet/checkpoints/f64-k3-lr0.0003-bs32-rtFalse-augTrueflip-rotate-a1-2-1-x-dcalpha\model-c378.sav'

model = keras.models.load_model(model_path)
assert isinstance(model, keras.Model)
model.summary()

data = tf.ones(shape=(100,300,300,1))

@tf.function
def predict(x):
  return model(x, training=False)
predict(data)
model(data, training=False)
# import timeit
# print("Time:", timeit.timeit(lambda: predict(data), number=100))

import time
import datetime
times = []
for _ in range(110):
  # start_time = time.time()
  start_time = datetime.datetime.now()
  model(data, training=False)
  end_time = datetime.datetime.now()
  times.append((end_time-start_time).microseconds)
  # print(end_time-start_time)

times = np.clip(times, np.percentile(times,10), np.inf)
print(times)
print(np.mean(times), np.std(times))