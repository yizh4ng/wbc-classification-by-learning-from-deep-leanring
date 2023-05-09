import pickle
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




# th.data_config = 'alpha'
th.data_config = 'beta'
th.data_dir = '../data'

train_set, test_set = du.load_data(th)



train_set = preprocess(train_set)
train_set, _ = train_set.split([50], ['train', 'test'], random=True,
                               over_key='cell_class')
test_set = preprocess(test_set)
train_set.report()
test_set.report()
# train_set = train_set.shuffle()[:20]
# train_set.view()
# quit()

TRANSFROM = [
  # Crop(x_range=[60, 240], y_range=[60, 240])
]

FEATURES = [
  Variance(),
  RegionSize(low=4),
  RegionSize(high=0.1),
  RegionSize(high=0.2), #
  RegionSize(high=3),
  RegionVariance(low=0.1), #
  RegionVariance(low=0.2),
  RegionVariance(low=2), #
  RegionVariance(low=3), #
  RegionVariance(high=0.1), #
  RegionVariance(high=0.2),
  RegionIntegrate(low=4),
  RegionIntegrate(low=3), #
  RegionIntegrate(low=2), #
  # RegionIntegrate(low=0.1), #
  RegionIntegrate(low=0),
  Maximum(),
]
# FEATURES = [
#   # Variance(),
#   RegionSize(low=4),
#   RegionSize(low=0.1),
#   RegionSize(low=0.2),
#   RegionSize(low=0.3),
#   # RegionSize(high=0.1),
#   # RegionSize(high=0.2), #
#   # RegionSize(high=3),
#   RegionVariance(low=0.1), #
#   RegionVariance(low=0.2),
#   RegionVariance(low=2), #
#   RegionVariance(low=3), #
#   # RegionVariance(high=0.1), #
#   # RegionVariance(high=0.2),
#   RegionIntegrate(low=4),
#   RegionIntegrate(low=3), #
#   RegionIntegrate(low=2), #
#   RegionIntegrate(low=0.1), #
#   # RegionIntegrate(low=0),
#   Maximum(),
# ]
# FEATURES = [
#   RegionSize(high=0.2),
#   RegionVariance(high=0.2), #
#   RegionIntegrate(high=0.2),
# ]
KEYS = [
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
  'features',
]



features = []
for transform in TRANSFROM:
  transfrom = transform(test_set)
  # feature_extractor(test_set)
  # features.append(feature)
  # feature_extractor.view_hist(train_set, key='cell_class')
  # feature_extractor.view_hist(test_set, key='cell_class')

train_features = []
test_features = []


def normalize(x):
  x -= np.mean(x)
  sigma = np.std(x)
  return x / sigma

for i, feature_extractor in enumerate(FEATURES):
  feature = feature_extractor(train_set, key='features')
  train_features.append(feature)

  # feature_extractor.view_hist(train_set, key='cell_class')
  feature = feature_extractor(test_set, key='features')
  test_features.append(feature)


#SVM
train_features = np.array(train_features)
test_features = np.array(test_features)

train_features = np.transpose(train_features, axes=(1, 0, 2))
test_features = np.transpose(test_features, axes=(1, 0, 2))

train_features = np.sum(train_features, axis=-1)
test_features = np.sum(test_features, axis=-1)

train_targets = np.array(train_set.properties['dense_labels'])
test_targets = np.array(test_set.properties['dense_labels'])

# highest=0
# best_modl = None
# for i in range(1000, 2000):
np.random.seed(0)
# index = np.arange(0,len(test_set))
# np.random.shuffle(index)
# features = features[index]
# targets = np.array(targets)[index]
# train_features = features[:80]
# trian_targets = np.array(targets)[:80]
# test_features = features[80:]
# test_targets = np.array(targets)[80:]

# SVM
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(gamma='scale',kernel='rbf', degree=3, C=1))

# Descion Tree
from sklearn import tree
# clf = tree.DecisionTreeClassifier(max_depth=5)

clf.fit(train_features, train_targets)

# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")

# with open('model.pkl', 'rb') as f:
#   clf = pickle.load(f)


prediction = clf.predict(train_features)

accracy = metrics.Accuracy()(prediction, train_targets)
print(accracy)

import datetime
test_data = np.ones(shape=(100, len(FEATURES)))
times = []
for _ in range(110):
  start = datetime.datetime.now()
  __ = clf.predict(test_data)
  end = datetime.datetime.now()
  times.append((end - start).microseconds)

times = np.clip(times, np.percentile(times,10), np.inf)
print(np.mean(times), np.std(times))
# quit()

prediction = clf.predict(test_features)

accracy = metrics.Accuracy()(prediction, test_targets)
print(accracy)
# if accracy.numpy() > highest:
#   print(accracy)
#   best = i
#   highest = accracy.numpy()
#   best_model = clf
#
# with open('model.pkl', 'wb') as f:
#   pickle.dump(best_model, f)

# print(best, highest)
# print(','.join(best_index.astype(str)))

# with open('model.pkl', 'rb') as f:
  # clf = pickle.load(f)
# prediction = clf.predict(test_features)

from tframe import console
cm = ConfusionMatrix(
  num_classes=2,
  class_names=['B', 'T'])
cm.fill(prediction, test_targets)
console.show_info('Confusion Matrix:')
console.write_line(cm.matrix_table(cell_width=4))
console.show_info('Evaluation Result:')
console.write_line(cm.make_table(decimal=4, class_details=True))
