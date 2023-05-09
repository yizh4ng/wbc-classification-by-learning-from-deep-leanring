import time
from lambo.zebravo.analyzer.analyzer import Analyzer
from lambo.zebravo.io_utili.saver import Saver

class Decoder(Analyzer):
  def __init__(self):
    super(Decoder, self).__init__()
    self.channel_num = 1

  def visualize(self, result):
    return result

  def save(self, result, path):
    Saver.save_image(path, self.frames, result, 'png-npy')