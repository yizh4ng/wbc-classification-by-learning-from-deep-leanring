from lambo.data_obj.interferogram import Interferogram
from lambo.zebra.decoder.decoder import Decoder

import numpy as np



class Subtracter(Decoder):

  def __init__(self, radius=80, boosted=False):
    # Call parent's constructor
    super(Subtracter, self).__init__()

    # Customized variables
    self.radius = radius
    self.boosted = boosted

  # region: Properties

  @Decoder.property()
  def background(self) -> Interferogram:
    bg = Interferogram(self.zinci.fetcher.background, radius=self.radius)
    if self.boosted: bg.booster = True
    return bg

  # endregion: Properties

  # region: Core Methods

  def _decode(self, x):
    ig = Interferogram(
      x, radius=self.radius)
    if self.boosted: ig.booster = True
    y = ig.extracted_angle_unwrapped
    ig.release()
    return y

  # endregion: Core Methods
