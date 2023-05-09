import numpy
import cupy as np
import time
# from scipy.signal import medfilt2d
from cupyx.scipy.ndimage import convolve
from ditto.image.interferogram import Interferogram
from lambo.zebravo.decoder.decoder import Decoder
from roma.ideology.noear import Nomear


off_set = 1


def get_image(x, radius=80,background_ig=None,boost=True, peak_index=None,
              mask=None):
  # ig = Interferogram(
  #   x, radius=radius)
  # if background_ig is not None:
  #   bg_ig = background_ig
  #   bg_ig.booster = boost
  #   ig._backgrounds = [bg_ig]
  #   ig.booster = boost
  #   y = ig.extracted_angle
  # else:
  #   y = ig.extracted_angle
  # ig.release()
  Fc = np.fft.fftshift(np.fft.fft2(x))
  if peak_index is None:
    Sc = np.abs(Fc)
    H, W = Fc.shape
    peak_mask = np.ones_like(x)
    h, w = peak_mask.shape
    # Put mask
    ci, cj = h // 2, w // 2
    d = int(0.05 * min(h, w))
    # mask[ci-d:ci+d, :] = 0
    peak_mask[ci - d:ci + d, cj - d:cj + d] = 0
    peak_mask[:, cj + d:] = 0
    region = Sc
    peak_index = np.unravel_index(np.argmax(region * peak_mask), region.shape)
    ci, cj = peak_index
    X, Y = np.ogrid[:H, :W]
    mask= np.sqrt((X - ci)**2 + (Y - cj)**2) <= radius

  masked = Fc * mask
  CI, CJ = [s // 2 for s in Fc.shape]
  pi, pj = peak_index
  # y = masked
  y = masked[(CI - pi) - radius: (CI - pi) + radius,
      (CJ - pj) - radius:(CJ - pj) + radius]
  # y = np.roll(masked, shift=np.array((CI - pi, CJ - pj)), axis=(0, 1))
  y = np.fft.ifft2(np.fft.ifftshift(y))
  return y



class DI(Decoder, Nomear):
  def __init__(self,radius, boost=False):
    super(DI, self).__init__()
    self.radius= radius
    self.boost = boost
    # self.thread_id = 2
    self.peak_index = None
    self.mask = None

  def get_angle(self, img, radius=80,background_ig=None,boost=False):
    self.img = np.array(img)
    def _get_angle():
      Fc = np.fft.fftshift(np.fft.fft2(np.array(img)))
      if self.peak_index is None or self.mask is None:
        Sc = np.abs(Fc)
        H, W = Fc.shape
        peak_mask = np.ones_like(img)
        h, w = peak_mask.shape
        # Put mask
        ci, cj = h // 2, w // 2
        d = int(0.05 * min(h, w))
        # mask[ci-d:ci+d, :] = 0
        peak_mask[ci - d:ci + d, cj - d:cj + d] = 0
        peak_mask[:, cj + d:] = 0
        region = Sc
        peak_index = np.unravel_index(np.argmax(region * peak_mask), region.shape)
        ci, cj = peak_index
        X, Y = np.ogrid[:H, :W]
        mask= np.sqrt((X - ci)**2 + (Y - cj)**2) <= radius
        self.peak_index = peak_index
        self.mask = mask

      masked = Fc * self.mask
      CI, CJ = [s // 2 for s in Fc.shape]
      pi, pj = self.peak_index
      # y = masked[-(CI - pi) - radius: -(CI - pi) + radius,
      #     -(CJ - pj) - radius:-(CJ - pj) + radius]
      y = np.roll(masked, shift=np.array((CI - pi, CJ - pj)), axis=(0, 1))
      if self.boost:
        y = y[CI-radius:CI+radius, CJ-radius:CJ+radius]
      y = np.fft.ifft2(np.fft.ifftshift(y))
      # for _ in range(10):
      #   y = uniform_filter(y, 3)
      y = np.angle(y)
      return  y
    return self.get_from_pocket('phase', initializer=_get_angle)

  def vertical_unwrap(self):
    def _unwrap():
      phase = np.unwrap(self.get_angle(self.img), axis=0)
      return phase
    return self.get_from_pocket('vertical_unwrapped_phase', initializer=_unwrap)

  def unwrap(self):
    def _unwrap():
      phase = np.unwrap(self.vertical_unwrap(), axis=1)
      return phase
    return self.get_from_pocket('unwrapped_phase', initializer=_unwrap)

  def jump_line_gradient(self):
    def _jump_line():
      phase = self.unwrap()
      line = phase[:, -5:-4]
      line = line[:-1] - line[1:]
      line[abs(line) < 1] = 0
      return line
    return self.get_from_pocket('jump_line', initializer=_jump_line)


  def horizontal_difference(self):
    def _horizontal_difference():
      lines = self.unwrap()
      lines = lines[:-1] - lines[1:]
      return lines
    return self.get_from_pocket('horizontal_difference', initializer=_horizontal_difference)

  def compensate(self):
    def _jump_points():
      gradient = self.jump_line_gradient()
      compen = np.concatenate([np.array([0]), np.cumsum(gradient)], axis=0)
      # compen = np.reshape(compen, (len(compen), 1))
      return compen
    return self.get_from_pocket('compensate', initializer=_jump_points)

  def jump_line_index(self):
    def _jump_line_index():
      return np.argwhere(np.abs(self.compensate()) > 0)
    return self.get_from_pocket('jump_index', initializer=_jump_line_index)

  def dual_phase(self):
    def _dual_phase():
      return self.unwrap() + np.expand_dims(self.compensate(), -1)
    return self.get_from_pocket('dual_phase', initializer=_dual_phase)

  def mean_jump_index(self):
    def _mean_jump_index():
      horiza_difference = self.unwrap()[:-1, :-2] - self.unwrap()[1:, :-2]
      jump_points =horiza_difference[:, :-1] - horiza_difference[:, 1:]
      return jump_points
    return self.get_from_pocket('dual_phase', initializer=_mean_jump_index)

  def compensated_phase(self):
    def _compensated_phase():
      _compensated_phase = np.copy(self.unwrap())
      # enchanced = self.unwrap() + self.dual_phase()
      horiza_difference = self.unwrap()[:-1, :-2] - self.unwrap()[1:, :-2]
      last_horiza_index = [[0]]
      for index in self.jump_line_index():
        index = index[0]
        horiza_index = np.argwhere(np.abs(horiza_difference[index - 1]) > 2)
        if len(horiza_index) == 0:
          print(index)
          horiza_index = last_horiza_index
        else:
          print(index, horiza_index[0][0])
          last_horiza_index = horiza_index
        _compensated_phase[index: index + 1, horiza_index[0][0]:] += self.compensate()[index]
      return _compensated_phase
    return self.get_from_pocket('compensated_phase', initializer=_compensated_phase)

  def decode(self,x, y):
    self.frame += 1
    self.release()
    self.get_angle(x, background_ig=y)
    # return self.compensated_phase().get()
    return self.dual_phase().get()


if __name__ == '__main__':
  from lambo.gui.vinci.vinci import DaVinci
  from ditto import VideoGenerator,Painter, ImageConfig, VideoConfig
  import cv2
  # p = VideoGenerator(ImageConfig, VideoConfig)
  p = Painter(ImageConfig)
  p.paint_samples()
  # bf = np.array(p.physics_based_fringe)
  bf = (cv2.imread('C:/projects/100007.tif'))[:,:,0]
  ig = Interferogram(bf,radius=80)
  bf = ig.high_frequency_filter
  bf = np.array(bf)
  di= DI(120, boost=False)
  da = DaVinci()
  di.get_angle(bf)
  max = np.maximum(di.unwrap(), di.dual_phase()).get(),
  min = np.minimum(di.unwrap(), di.dual_phase()).get()
  da.objects = [
    di.get_angle(bf).get(),
    di.unwrap().get(),
    # di.jump_line_index().get(),
    di.dual_phase().get(),
    np.unwrap(di.dual_phase().get(), axis=1).get(),
    # di.horizontal_difference().get(),
    # di.compensated_phase().get()
  ]
  da.add_plotter(da.imshow)
  da.show()
