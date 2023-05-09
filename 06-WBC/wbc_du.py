from wbc.wbc_agent import WBCSet, WBCAgent
from tframe.data.dataset import DataSet


def load_data(th):
  trainset, testset  = WBCAgent.load(th)
  return trainset, testset
