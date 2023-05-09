from tframe.data.dataset import DataSet
from lambo.gui.vinci.vinci import DaVinci
import numpy as np
from tframe import pedia



class WBCSet(DataSet):
  def __init__(self, **kwargs):
    super(WBCSet, self).__init__(**kwargs)

  def view(self, key='features'):
    da = DaVinci()
    da.objects = self.data_dict[key]
    # one_hot
    targets = self.targets
    if len(np.array(self.targets).shape) != 1:
      targets = []
      for target in self.targets:
        label = np.sum(np.argwhere(target == 1))
        targets.append(self.properties[pedia.classes][label])
    da.object_titles = targets
    da.add_plotter(da.imshow)
    da.show()