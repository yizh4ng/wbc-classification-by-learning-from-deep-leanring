from lambo.zebravo.io_utili.fetch import Fetcher
from lambo.zebravo.io_utili.drivers.flir_api import FLIRCamDev
import time


class FLIRFetcher(Fetcher):

  def __init__(self, max_len=20, radius=80):
    super(FLIRFetcher, self).__init__(max_len=max_len, radius=radius)

    self.camera = FLIRCamDev()
    # self.camera.set_frame_rate(60)


  def _init(self):
    self.camera.start()
    self.camera.set_buffer_count(1)


  def _loop(self):
    im = self.camera.read().GetNDArray()

    self.append_to_buffer(im)
    time.sleep(0)


  def _finalize(self):
    self.camera.stop()
    self.camera.close()
