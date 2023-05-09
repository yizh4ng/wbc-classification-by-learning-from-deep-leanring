from lambo.data_obj.cu_interferogram import Interferogram
from lambo.zebravo.decoder.decoder import Decoder

class Subtractor(Decoder):
  def __init__(self, radius=80, boost=False, invert=False):
    super(Subtractor, self).__init__()
    self.radius = radius
    self.boost = boost
    self.invert = invert

  def analyze(self,davincv):
    x = davincv.objects[davincv.raw_channel]
    background_ig = davincv.background_ig
    ig = Interferogram(
      x, radius=self.radius, boost=self.boost, invert=self.invert)
    # ig.booster = self.boost
    if background_ig is not None:
      bg_ig = background_ig
      assert isinstance(bg_ig, Interferogram)
      bg_ig.booster = ig.booster
      # print(bg_ig.booster)
      ig._backgrounds = [bg_ig]
      # ig.booster = ig.booster
      y = ig.unwrapped_phase
      # y = ig.flattened_phase
    else:
      # ig.booster = self.boost
      y = ig.extracted_angle_unwrapped
    ig.release()
    return y

