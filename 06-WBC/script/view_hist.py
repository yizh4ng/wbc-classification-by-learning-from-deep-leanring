import pickle

import matplotlib
from matplotlib.ticker import StrMethodFormatter
from sklearn.linear_model import SGDClassifier

from tframe.utils.maths.confusion_matrix import ConfusionMatrix
from wbc_core import th, preprocess
from tframe.data.feature_extractor.cell import *
import wbc_du as du
import numpy as np
from sklearn import svm
from tframe.quantity import *
import tensorflow.keras.metrics as metrics
from tframe.data.feature_extractor.base import FeatureExtractor

matplotlib.rc('font', size=20, family="Times New Roman")

# th.data_config = 'alpha'
th.data_config = 'beta'
th.data_dir = '../data'

train_set, test_set = du.load_data(th)


train_set = preprocess(train_set)
# test_set = preprocess(test_set)
# train_set = train_set.shuffle()[:20]
# train_set.view()
# quit()

TRANSFROM = [
  # Crop(x_range=[60, 240], y_range=[60, 240])
]

FEATURES = [
  Variance(),
  # RegionSize(low=4),
  # RegionSize(high=0.1),
  RegionSize(high=3),
  # RegionVariance(low=0.1),
  # RegionIntegrate(low=4),
  # RegionIntegrate(low=0),
  Maximum(),
]

KEYS = [
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  # 'Crop region',
  # 'Crop region',
  'features'
]



features = []
for transform in TRANSFROM:
  transfrom = transform(train_set)
  # feature_extractor(test_set)
  # features.append(feature)
  # feature_extractor.view_hist(train_set, key='cell_class')
  # feature_extractor.view_hist(test_set, key='cell_class')

for i, feature_extractor in enumerate(FEATURES):
  plt.figure(figsize=(9, 6))
  plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
  feature = feature_extractor(train_set)
  features.append(feature)
  feature_extractor.view_hist(train_set, key='cell_class', save_path=r'C:\Users\Administrator\Dropbox\YiZhang\Manuscript\Support Vector Machine based white blood cell classification by learning from deep learning\figures\figure3')
