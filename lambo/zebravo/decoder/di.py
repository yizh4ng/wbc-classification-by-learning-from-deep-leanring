from scipy.signal import hilbert
import numpy
import cupy as np
import time
# from scipy.signal import medfilt2d
from cupyx.scipy.ndimage import median_filter, minimum_filter, maximum_filter, uniform_filter, gaussian_filter
from ditto.image.interferogram import Interferogram
from lambo.zebravo.decoder.decoder import Decoder

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



def gradient(extracted_angle):
  x_gradient = extracted_angle[:-off_set, :] - extracted_angle[off_set:, :]
  y_gradient = extracted_angle[:, :-off_set] - extracted_angle[:, off_set:]
  return x_gradient, y_gradient

def correct_gradient(gradient):
  magic = 3
  gradient[gradient - 2 * np.pi > -magic] -= 2 * np.pi
  gradient[2 * np.pi + gradient < magic] += 2 * np.pi
  # gradient[gradient - 2 * np.pi > -magic] = 0
  # gradient[2 * np.pi + gradient < magic] = 0
  return gradient

def delete_outlier_gradient(gradient, direction='x'):
  # gradient = np.array(gradient)
  # sd = np.std(gradient)
  # outlier_index = np.argwhere(np.abs(gradient) > 10 * sd)
  # for index in outlier_index:
  #   i, j = index[0], index[1]
  #   r = 2
  #   if gradient[i,j] < 0:
  #     if direction =='x':
  #       gradient[i,j] = np.max(gradient[i:i+1, j - r:j])
  #   else:
  #     if direction =='x':
  #       gradient[i,j] = np.min(gradient[i:i+1, j - r:j])
  gradient = median_filter(gradient, size=3)
  return gradient


def decode(x, radius=80, background_ig=None,boost=False):
  x = np.array(x)

  # tic = time.time()
  extracted_image = get_image(x, radius, background_ig,boost=boost)
  extracted_image /= background_ig.extracted_image
  extracted_angle = np.angle(extracted_image)

  # print(time.time() - tic)
  # tic = time.time()
  x_gradient, y_gradient = gradient(extracted_angle)

  x_gradient = delete_outlier_gradient(x_gradient)
  y_gradient = delete_outlier_gradient(y_gradient)

  x_gradient = correct_gradient(x_gradient)
  y_gradient = correct_gradient(y_gradient)


  x = np.cumsum(np.concatenate([extracted_angle[:1, :], x_gradient], axis=0), axis=0)
  y = np.cumsum(np.concatenate([extracted_angle[:, :1], y_gradient], axis=1), axis=1)

  # print(time.time() - tic)
  return (-x - y).get()

class DI(Decoder):
  def __init__(self,radius, boost=False):
    super(DI, self).__init__()
    self.radius= radius
    self.boost = boost
    # self.thread_id = 2
    self.peak_index = None
    self.mask = None

  def _get_angel(self, x, radius=80,background_ig=None,boost=False):

    Fc = np.fft.fftshift(np.fft.fft2(x))
    if self.peak_index is None or self.mask is None:
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
    return y

  def _decode_angle(self,x, y):
    x = np.array(x)

    # tic = time.time()
    extracted_angle = self._get_angel(x, self.radius, y, boost=self.boost)
    return extracted_angle

  def differentiate_integrate(self, extracted_angle):
    # print(time.time() - tic)
    # tic = time.time()
    x_gradient, y_gradient = gradient(extracted_angle)

    x_gradient = correct_gradient(x_gradient)
    y_gradient = correct_gradient(y_gradient)

    x_gradient = delete_outlier_gradient(x_gradient)
    y_gradient = delete_outlier_gradient(y_gradient)

    x = np.cumsum(np.concatenate([np.zeros_like(extracted_angle[:1, 2:-2]), x_gradient[2:-2, 2:-2]], axis=0),
                  axis=0)
    y = np.cumsum(np.concatenate([np.zeros_like(extracted_angle[2:-2, :1]), y_gradient[2:-2, 2:-2]], axis=1),
                  axis=1)
    if off_set == 1:
      return (- x[:, :] - y[:, :])
    return (- x[:, :-off_set+1] - y[:-off_set + 1. :])


  def decode(self,x, y):
    self.frame += 1
    # phase = np.unwrap(np.unwrap(self._decode_angle(x, y), axis=0), axis=1)
    # return phase

    return self.differentiate_integrate(self._decode_angle(x, y)).get()


if __name__ == '__main__':
  import copy
  from lambo.gui.vinci.vinci import DaVinci
  from ditto import VideoGenerator,Painter, ImageConfig, VideoConfig
  import cv2
  # p = VideoGenerator(ImageConfig, VideoConfig)
  p = Painter(ImageConfig)
  p.paint_samples()
  # bf = np.array(p.physics_based_fringe)
  bf = (cv2.imread('G:/projects/data_old/05-bead/100007.tif'))[:,:,0]
  ig = Interferogram(bf,radius=80)
  bf = ig.high_frequency_filter
  bf = np.array(bf)
  di= DI(120, boost=False)
  angle = di._get_angel(bf)
  # phase = di.decode(bf, None)
  # alpha = 0.1
  # phase = np.unwrap(angle, alpha * np.pi, axis=1)
  # phase = np.unwrap(phase, axis=0, discont=alpha * np.pi)
  # gx, gy = gradient(phase)
  # cgx, cgy = correct_gradient(np.copy(gx)), correct_gradient(np.copy(gy))
  # dgx = delete_outlier_gradient(np.copy(cgx))
  da = DaVinci()
  da.objects = [
                # gx.get(),
                # gy.get(),
                # phase.get(),
                di.decode(bf,None)
               ]
  da.add_plotter(da.imshow)
  da.show()
