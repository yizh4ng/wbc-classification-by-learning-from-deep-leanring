import numpy as np
from skimage.restoration import unwrap_phase
from lambo.data_obj.interferogram import Interferogram
from roma.ideology.noear import Nomear
from lambo.gui.vinci.vinci import DaVinci



def read_npz(npz):
  npz = np.load(npz)
  return npz[npz.files[0]].tolist()

def pad(im, shape):
  resize = np.array(shape) - np.array(im).shape
  if any(resize <= 0):
    return im
  else:
    im = np.pad(im,((resize[0]//2, resize[0]//2),
                    (resize[1]//2, resize[1]//2)),
                'constant', constant_values=(0, 0))
    return im

def SVD_compressed(img, k):
  U, sigma, V = np.linalg.svd(np.real(img))
  real = np.matrix(U[:, :k]) @ np.diag(sigma[:k]) @ np.matrix(V[:k, :])
  U, sigma, V = np.linalg.svd(np.imag(img))
  imag = np.matrix(U[:, :k]) @ np.diag(sigma[:k]) @ np.matrix(V[:k, :])
  img =  real + 1j * imag
  return np.array(img)


class Reader(Nomear):
  def __init__(self, resize=None, k_space_size=None, first_k_feature=None):
    super(Reader, self).__init__()
    self.resize = resize
    self.k_space_size = k_space_size
    self.first_k_feature = first_k_feature
    self.blind_mask = None

  def set_background(self, back_ground):
    self.release()
    # if 'background' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('background')
    self.put_into_pocket('background', back_ground)
    # if 'padded_background' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('padded_background')
    # if 'ifft_padded_background' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('ifft_padded_background')
    # if 'result_phase' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('result_phase')

  def set_image(self, img):
    back_ground = self.background
    self.release()
    self.put_into_pocket('background', back_ground)
    # if 'image' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('image')
    self.put_into_pocket('image', img)
    # if 'padded_image' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('padded_image')
    # if 'result_phase' in self._cloud_pocket.keys():
    #   self._cloud_pocket.pop('result_phase')

  def set_bind_mask(self, coords):
    if len(coords) != 2:
      # print('blind mask cancelled')
      back_ground = self.background
      image = self.image
      self.release()
      self.put_into_pocket('background', back_ground)
      self.put_into_pocket('image', image)
      self.blind_mask = None
    else:
      back_ground = self.background
      image = self.image
      self.release()
      self.put_into_pocket('background', back_ground)
      self.put_into_pocket('image', image)
      ci, cj = center = coords[0]
      radius = np.linalg.norm(np.array(coords[1]) - np.array(coords[0]))
      if self.resize is not None:
        H, W = self.resize
      else:
        H, W = self.image.shape

      X, Y = np.ogrid[:H, :W]
      mask = np.sqrt((X - ci)**2 + (Y - cj)**2) >= radius
      self.blind_mask = mask
      # print('blind mask set')

  @property
  def background(self):
    back_ground = np.array(self.get_from_pocket('background'))

    if self.k_space_size is not None:
      H, W = back_ground.shape
      CI, CJ = [s // 2 for s in back_ground.shape]
      X, Y = np.ogrid[:H, :W]
      mask = np.sqrt((X - CI)**2 + (Y - CJ)**2) <= self.k_space_size
      back_ground = mask * back_ground


    return back_ground

  @property
  def image(self):
    image = np.array(self.get_from_pocket('image'))
    if self.k_space_size is not None:
      H, W = image.shape
      CI, CJ = [s // 2 for s in image.shape]
      X, Y = np.ogrid[:H, :W]
      mask = np.sqrt((X - CI)**2 + (Y - CJ)**2) <= self.k_space_size
      image = mask * image
    return image

  @property
  def padded_background(self):
    def _padded_background():
      background = self.background
      k = self.first_k_feature
      # print(k)
      if k is not None:
        background = SVD_compressed(background, k)
      if self.resize is not None:
        background =  pad(background, self.resize)
      if self.blind_mask is not None:
        background = self.blind_mask * background
        # print('hi')
      return background
    return self.get_from_pocket('padded_background',
                                initializer=_padded_background)

  @property
  def ifft_padded_background(self):
    return self.get_from_pocket('ifft_padded_background',
                              initializer=lambda: np.fft.ifft2(self.padded_background))

  @property
  def padded_background_phase(self):
    return self.get_from_pocket('padded_background_pahse',
                                initializer=lambda :  -unwrap_phase(np.angle(self.ifft_padded_background)))

  @property
  def padded_image(self):
    def _image():
      image = self.image
      k = self.first_k_feature
      # print(k)
      if k is not None:
        image = SVD_compressed(image, k)
      if self.resize is not None:
        image = pad(image, self.resize)
      if self.blind_mask is not None:
        # print('hi')
        image = self.blind_mask * image
      return image
    return self.get_from_pocket('padded_image',
                                initializer=_image)

  @property
  def ifft_padded_image(self):
    return self.get_from_pocket('ifft_padded_image',
                                initializer=lambda: np.fft.ifft2(self.padded_image))

  @property
  def padded_image_phase(self):
    return self.get_from_pocket('padded_image_pahse',
                                initializer=lambda :  -unwrap_phase(np.angle(self.ifft_padded_image)))

  @property
  def corrected_ifft(self):
    return self.get_from_pocket('corrected_ifft', initializer=lambda : np.divide(
      self.ifft_padded_image, self.ifft_padded_background, out=np.zeros_like(
        self.ifft_padded_image
      ), where=self.ifft_padded_background!=0
    ))

  @property
  def corrected_ifft_phase(self):
    return self.get_from_pocket('corrected_ifft_phase', initializer= lambda :-unwrap_phase(np.angle(
      self.corrected_ifft
    )))

  def decode(self):
    def phase():
      # a = np.fft.ifft2(self.padded_image)
      # b = self.ifft_padded_background
      # c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
      # phase = -unwrap_phase(np.angle(self.corrected_ifft))
      phase = self.corrected_ifft_phase
      ig = Interferogram(img=np.array([0]))
      ig.put_into_pocket('unwrapped_phase', phase)
      result = ig.flattened_phase
      ig.release()
      return result
    return self.get_from_pocket('result_phase', initializer=lambda : phase())
    # return result

class ReaderVisualizer(DaVinci):
  def __init__(self):
    super(ReaderVisualizer, self).__init__()
    self.alpha = 500
    self.canvas.mpl_connect('button_press_event', self.select_point)
    self.state_machine.register_key_event('c', self.cancell_selected_points)
    self.state_machine.register_key_event('d', self.cancell_selected_points)
    self.state_machine.register_key_event('b', self.delete_selected_area)
    self.chosen_points = []


  def select_point(self, event):
    ix, iy = event.xdata, event.ydata
    self.chosen_points.append([iy, ix])
    print(f'Select: {iy, ix}')

  def cancell_selected_points(self):
    print('Cancell selected points')
    self.chosen_points = []

  def delete_selected_area(self):
    for object in self.objects:
      assert isinstance(object, Reader)
      object.set_bind_mask(self.chosen_points)
    self.chosen_points = []

  def vis_complex(self, x):
    return self.alpha * np.abs(x) / np.max(np.abs(x))

  def set_alpha(self, alpha:int):
    self.alpha = alpha

  def decode(self, x):
    return -unwrap_phase(np.angle(np.fft.ifft2(x)))

  def show_image_k_space(self, x):
    assert isinstance(x, Reader)
    self.imshow(self.vis_complex(x.padded_image), cmap='gray')

  def show_background_k_space(self, x):
    assert isinstance(x, Reader)
    self.imshow(self.vis_complex(x.padded_background), cmap='gray')

  def show_image_phase(self, x):
    assert isinstance(x, Reader)
    self.imshow(x.padded_image_phase)

  def show_background_phase(self, x):
    assert isinstance(x, Reader)
    self.imshow(x.padded_background_phase)

  def show_image_ifft(self, x):
    assert isinstance(x, Reader)
    self.imshow(self.vis_complex(x.ifft_padded_image))

  def show_background_ifft(self, x):
    assert isinstance(x, Reader)
    self.imshow(self.vis_complex(x.ifft_padded_background))

  def show_corrected_ifft(self, x):
    assert isinstance(x, Reader)
    self.imshow(self.vis_complex(x.corrected_ifft))

  def show_corrected_ifft_phase(self, x):
    assert isinstance(x, Reader)
    self.imshow(x.corrected_ifft_phase)

  def show_reconstruction(self, x):
    assert isinstance(x, Reader)
    self.imshow(x.decode())

if __name__ == '__main__':
  from roma import console
  import os
  # resize = (1024, 1280)
  resize = None
  # SAMPLE_PATH = r'C:\Users\Administrator\Desktop\Merck Data\microfluidics imaging sample\17-21-31-.730684 host cells'
  # SAMPLE_PATH = r'E:\projects\lambai\lambo\zebravo\sample\16-18-29-.838323'
  # SAMPLE_PATH = r'C:\Users\Administrator\Desktop\Merck Data\microfluidics imaging sample\17-25-06-.867615 host cells'
  # SAMPLE_PATH = r'C:\Users\Administrator\Desktop\Merck Data\microfluidics imaging sample\18-04-31-.550981 high titer'
  SAMPLE_PATH = r'C:\Users\Administrator\Desktop\Merck Data\microfluidics imaging sample\18-11-01-.633662 high titer'
  # SAMPLE_PATH = r'G:\Merck_data\14-39-55-.228206 low titer 0307'
  # SAMPLE_PATH = r'G:\Merck_data\15-58-44-.043009 low titer'
  # SAMPLE_PATH = r'G:\Merck_data\14-52-20-.149417 high titer 0307'
  # SAMPLE_PATH = r'G:\Merck_data\15-20-01-.196822 high titer 0307'
  test_num = 100
  k_space_size = 30
  console.show_status('Reading samples...')
  path = os.walk(SAMPLE_PATH)

  back_ground = read_npz(SAMPLE_PATH + '/background/background.npz')
  rv = ReaderVisualizer()

  # for root, directories, samples_path in path:
  #   image_cursor = 1
  #   for i, sample_path in enumerate(samples_path):
  #     if i > (test_num // 200) + 1: break
  #     if sample_path == 'test_save.py' or sample_path.split('.')[
  #       0] == 'background': continue
  #     sample = read_npz(SAMPLE_PATH + '/' + sample_path)
  #
  #     for j in range(len(sample)):
  #       if j > test_num - i * 200: break
  #       reader = Reader(resize, k_space_size=k_space_size)
  #       reader.set_background(back_ground=back_ground)
  #       reader.set_image(sample[j])
  #       rv.objects.append(reader)

  for root, directories, samples_path in path:
    image_cursor = 1
    for i, sample_path in enumerate(samples_path):
      if i > (test_num // 200) + 1: break
      if sample_path == 'test_save.py' or sample_path.split('.')[
        0] == 'background': continue
      sample = read_npz(SAMPLE_PATH + '/' + sample_path)

      # for k in np.arange(5, 80, 5):
      #   reader = Reader(resize, k_space_size=k)
      #   reader.set_background(back_ground=back_ground)
      #   reader.set_image(sample[0])
      #   rv.objects.append(reader)
      for k in np.arange(1, 80, 1):
        reader = Reader(resize, first_k_feature=k)
        reader.set_background(back_ground=back_ground)
        reader.set_image(sample[0])
        rv.objects.append(reader)
      break
    break

  rv.add_plotter(rv.show_image_k_space)
  rv.add_plotter(rv.show_image_ifft)
  rv.add_plotter(rv.show_image_phase)
  rv.add_plotter(rv.show_background_k_space)
  rv.add_plotter(rv.show_background_ifft)
  rv.add_plotter(rv.show_background_phase)
  rv.add_plotter(rv.show_corrected_ifft)
  rv.add_plotter(rv.show_corrected_ifft_phase)
  rv.add_plotter(rv.show_reconstruction)
  rv.show()