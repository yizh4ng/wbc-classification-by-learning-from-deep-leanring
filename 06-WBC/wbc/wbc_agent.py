from wbc.wbc_set import WBCSet
from tframe import pedia
import os
from roma import console
import roma.spqr.finder as finder
import numpy as np

from tframe.data.prepocess.images import rotagram


class WBCAgent():
  @classmethod
  def get_data_name(cls, th, prefix):
    if 'new' in th.raw_data_dir:
      return '-'.join([prefix, 'rog{}'.format(th.rotagram),
                       th.feature_key]) + 'new.tfd'
    else:
      return '-'.join([prefix, 'rog{}'.format(th.rotagram),
                      th.feature_key]) + '.tfd'

  @classmethod
  def load(cls, th):
    train_set, test_set = cls.load_as_tframe_data(th)
    return train_set, test_set


  @classmethod
  def load_as_tframe_data(cls, th):
    expected_train_file_path = os.path.join(th.data_dir, WBCAgent.get_data_name(th, 'train'))
    expected_test_file_path = os.path.join(th.data_dir, WBCAgent.get_data_name(th, 'test'))
    if os.path.exists(expected_train_file_path) \
        and os.path.exists(expected_test_file_path):
      console.show_status('Loading existing datasets...')

      return WBCSet.load(expected_train_file_path), \
             WBCSet.load(expected_test_file_path)

    if th.rotagram == True:
      th.rotagram = False
      train_set, test_set  = WBCAgent.load_as_tframe_data(th)
      train_set.features = rotagram(train_set.features, verbose=True)
      test_set.features = rotagram(test_set.features, verbose=True)
      th.rotagram = True
    else:
      train_set, test_set = cls.load_as_numpy_arrays(th)

    # Save data_set
    console.show_status('Saving dataset ...')
    train_set.save(expected_train_file_path)
    console.show_status('Trainset has been saved to {}'.format(expected_train_file_path))
    test_set.save(expected_test_file_path)
    console.show_status('Testset has been saved to {}'.format(expected_test_file_path))
    return train_set, test_set


  @classmethod
  def load_as_numpy_arrays(cls, th):
    raw_data_dir = th.raw_data_dir
    FEATURES = ['phase', 'amplitude']
    assert th.feature_key in FEATURES
    feature_index = FEATURES.index(th.feature_key)
    CLASSES = ['B', 'T', 'Monocyte', 'Granulocyte', 'CD4', 'CD8']

    train_features_file = 'train_data_set.pickle'
    train_target_file = 'train_target_set.pickle'
    test_features_file = 'test_data_set.pickle'
    test_target_file = 'test_target_set.pickle'

    trainset = WBCSet()
    dense_labels = np.array(np.load(os.path.join(raw_data_dir, train_target_file), allow_pickle=True))
    cell_class = [CLASSES[i] for i in dense_labels]
    features = np.array(np.load(os.path.join(raw_data_dir, train_features_file), allow_pickle=True))
    trainset.features = features[:,:,:,feature_index:feature_index+1]
    trainset.properties.update({ 'cell_class': cell_class})

    testset = WBCSet()
    dense_labels = np.array(np.load(os.path.join(raw_data_dir, test_target_file), allow_pickle=True))
    cell_class = [CLASSES[i] for i in dense_labels]
    features = np.array(np.load(os.path.join(raw_data_dir, test_features_file), allow_pickle=True))
    testset.features = features[:,:,:,feature_index:feature_index+1]
    testset.properties.update({ 'cell_class': cell_class})

    return trainset, testset



if __name__ == '__main__':
  pass
