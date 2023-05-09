import sys, os
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)

print(sys.path)
from wbc_core import th, du
from tframe.data.feature_extractor.cell import *
from tensorflow import keras
import tensorflow as tf


th.data_config = 'alpha'
# th.data_dir = '/home/yiz/share/projects/lambai_v2/06-WBC/data'
# model_path =r'/home/yiz/share/projects/lambai_v2/06-WBC/resnet/checkpoints/f1-k3-lr0.0003-bs32-rtFalse-augTrue-a1-1-x-dcalpha/model-c4320.sav'
th.data_dir = '../data'
model_path = '../resnet/checkpoints/f1-k3-lr0.0003-bs32-rtFalse-augTrueflip-rotate-a1-1-x-dcalpha/model-c4320.sav'

FEATURES = [
  # RegionSize(low=0.2),
  Variance(),
  Maximum(),
]


model = keras.models.load_model(model_path)


feature_layer = model.layers[-2].output

model_temp = tf.keras.Model(inputs=model.input, outputs=feature_layer)
# model_temp.summury()

train_set, _ = du.load_data(th)
train_set, _ = train_set.split([128], names=['train_sampled', 'remaining'], over_key='cell_class')
train_set.report()
print('features')
features = []
for i, feature_extractor in enumerate(FEATURES):
  feature = feature_extractor(train_set)
  features.append(feature)

# print(features)
features  = -np.sum(features, axis=-1)

print('model features')
model_features = model_temp(train_set.features).numpy()
model_features  = np.transpose(model_features)
# print(model_features)

print(np.corrcoef(features[0], model_features[0]))
print(np.corrcoef(features[1], model_features[1]))







