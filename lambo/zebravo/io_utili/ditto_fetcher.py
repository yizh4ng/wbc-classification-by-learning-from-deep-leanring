from roma import console
from ditto import VideoGenerator, VideoConfig, ImageConfig
import time
from readerwriterlock import rwlock
from lambo.zebravo.io_utili.fetch import Fetcher

marker = rwlock.RWLockFair()

class DittoFetcher(Fetcher):

  def __init__(self, fps=200, max_len=20, L=None):
    super(DittoFetcher, self).__init__(fps, max_len, L)


  def _init(self):
    # Fetch all interferograms
    console.show_status(f'Preparing data...')
    Config = VideoConfig
    Config['video_length'] = self.max_len
    vg = VideoGenerator(ImageConfig, VideoConfig)
    vg.generate()
    self.interferograms = vg.fringe_stack

    assert self.fps > 0
    self.index = 0
    console.show_status('Looping ...')

  def _loop(self):
    index = self.index % len(self.interferograms)
    self.append_to_buffer(self.interferograms[index])
    time.sleep(0)
    self.index += 1

if __name__ == '__main__':
  from lambo.data_obj.interferogram import Interferogram
  from lambo.zebravo.gui.zincv import Zincv
  import numpy as np
  df = DittoFetcher()
  davincv = Zincv()
  davincv.pause = False
  davincv.background_ig = Interferogram(np.ones((500,500)), radius=80)
  davincv.recording = True
  davincv.save_path = './test'
  df.fetch(davincv, True)
