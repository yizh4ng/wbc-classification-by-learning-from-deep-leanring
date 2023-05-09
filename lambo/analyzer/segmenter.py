import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed



from roma.ideology.noear import Nomear
from lambo.gui.vinci.vinci import DaVinci



class Segmenter(Nomear):

  def __init__(self, img, threshold, erode_dilate_kenerl_size=7,
               erode_num=2, dilate_num=2, min_size=500, max_size=5000,
               save_size=150, **kwargs):
    # put the image into local pocket
    self.put_into_pocket('image', img, local=True)

    # put config into cloud pocket
    self.put_into_pocket('threshold', threshold)
    self.put_into_pocket('erode_dilate_kenerl_size', erode_dilate_kenerl_size)
    self.put_into_pocket('erode_num', erode_num)
    self.put_into_pocket('dilate_num', dilate_num)
    self.put_into_pocket('min_size', min_size)
    self.put_into_pocket('max_size', max_size)
    self.put_into_pocket('save_size', save_size)
    for k in kwargs:
      self.put_into_pocket(k, kwargs[k])

  @property
  def image(self):
    return self.get_from_pocket(key='image', local=True)

  @property
  def clipped_image(self):
    def _clipped_image():
      assert len(self['threshold']) == 2
      return np.clip(self.image, *self['threshold'])
    return self.get_from_pocket('clipped_image', initializer=_clipped_image, local=True)

  @property
  def normalized_image(self):
    def _normalized_image():
      image = self.clipped_image.copy()
      cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
      image = image.astype('uint8')
      return image
    return self.get_from_pocket('normalized_image', initializer=_normalized_image, local=True)

  @property
  def erode_dilate(self):
    def _erode_dilate():
      image = self.normalized_image
      kernel = np.ones([self['erode_dilate_kenerl_size']] * 2, np.uint8)
      image = cv2.erode(image, kernel, iterations=self['erode_num'])
      image = cv2.dilate(image, kernel, iterations=self['dilate_num'])
      return image
    return self.get_from_pocket('erode_dilate', initializer=_erode_dilate, local=True)

  @property
  def connected_area(self):
    def _connected_area():
      image = self.erode_dilate
      nb_componets, output, stats, _ = cv2.connectedComponentsWithStats(
        image, connectivity=8)
      sizes = stats[1:, -1]
      nb_componets = nb_componets - 1
      connectted_area = np.zeros((image.shape), dtype=np.uint8)
      for i in range(0, nb_componets):
        if sizes[i] >= self['min_size'] and sizes[i] <= self['max_size']:
          connectted_area[output == i + 1] = 255
      return connectted_area
    return self.get_from_pocket('connect_area', initializer=_connected_area, local=True)

  @property
  def distance(self):
    def _distance():
      image = self.connected_area
      distance = ndi.distance_transform_edt(image)
      # distance = cv2.GaussianBlur(distance, (5,5), 0)
      norm_dist=np.zeros(distance.shape)
      cv2.normalize(distance, norm_dist, 0, 1, cv2.NORM_MINMAX)
      return norm_dist
    return self.get_from_pocket('distance', initializer=_distance, local=True)

  @property
  def watershed_label(self):
    def _watershed_label():
      norm_dist = self.distance
      mask = norm_dist
      markers, _ = ndi.label(mask)
      labels = watershed(-norm_dist, markers, mask=self.connected_area)
      return labels
    return self.get_from_pocket('watershed_label', initializer=_watershed_label, local=True)

  @property
  def segmentation_scheme(self):
    def _segmentation_scheme():
      labels = self.watershed_label
      image = self.image.copy()
      cell_num = np.max(labels)
      # bounding_boxes = self.bounding_boxes
      for i, cell in enumerate(range(cell_num)):
        cell_area = (labels == cell+1)
        # contours2, _ = cv2.findContours(int(255)*cell_area, cv2.RETR_CCOMP,
        #                                 cv2.CHAIN_APPROX_SIMPLE)
        if np.sum(cell_area) > 0:
          cell_coords=np.argwhere(cell_area)
          x_min,y_min=cell_coords.min(axis=0)
          x_max,y_max=cell_coords.max(axis=0)
          if len(self.image.shape) == 3:
            cv2.putText(image, f'{i}', (int((y_max+y_min)/2),int((x_max+x_min)/2)), color=(0,0,255),
                        thickness=2, fontScale=1, fontFace=cv2.LINE_AA)
          else:
            cv2.putText(image, f'{i}',
                        (int((y_max + y_min) / 2), int((x_max + x_min) / 2)),
                        color=1,
                        thickness=2, fontScale=1, fontFace=cv2.LINE_AA)
      return image
    return self.get_from_pocket('segmentation_scheme', initializer=_segmentation_scheme, local=True)

  @property
  def labels(self):
    def _labels():
      results = []
      labels = self.watershed_label
      cell_num = np.max(labels)
      for cell in range(cell_num):
        cell_area = (labels == cell + 1)
        if np.sum(cell_area) > 0:
         results.append(labels == cell + 1)
      return results
    return self.get_from_pocket('labels', initializer=_labels, local=True)

  @property
  def bounding_boxes(self):
    def _bounding_boxes():
      field_of_interest = []
      labels = self.watershed_label
      cell_num = np.max(labels)
      for cell in range(cell_num):
        cell_area = (labels == cell + 1)
        if np.sum(cell_area) > 0:
          cell_coords = np.argwhere(cell_area)
          x_min, y_min = cell_coords.min(axis=0)
          x_max, y_max = cell_coords.max(axis=0)
          center_of_cell = [(x_min + x_max) / 2, (y_min + y_max) / 2]
          save_size = self['save_size']
          [x_min_cell, y_min_cell] = np.array([max(0, center_of_cell[0] - save_size / 2),
                                      max(0, center_of_cell[1] - save_size / 2)]).astype(int)
          [x_max_cell, y_max_cell] = np.array([min(x_min_cell + save_size, self.image.shape[0]),
                                      min(y_min_cell + save_size, self.image.shape[1])]).astype(int)
          field_of_interest.append([[x_min_cell,y_min_cell], [x_max_cell, y_max_cell]])
      return field_of_interest
    return self.get_from_pocket('bounding_boxes', initializer=_bounding_boxes, local=True)

  @property
  def segmented_images(self):
    def _segmented_images():
      segmented_images = []
      field_of_interets = self.bounding_boxes
      for j, field_of_interet in enumerate(field_of_interets):
        field_of_interet = np.array(field_of_interet).astype(int)
        x_min, y_min = field_of_interet[0]
        x_max, y_max = field_of_interet[1]
        individual_object = self.image[x_min:x_max, y_min:y_max]
        segmented_images.append(individual_object)
      return segmented_images
    return self.get_from_pocket('segmented_images', initializer=_segmented_images, local=True)

  @property
  def fake_labels(self):
    def _fake_labels():
      field_of_interest = []
      labels = self.watershed_label
      cell_num = np.max(labels)
      results = np.zeros_like(self.image)
      for cell in range(cell_num):
        cell_area = (labels == cell + 1)
        # if cell in (1,3):
        #   results[cell_area] = 255
        # else:
        results[cell_area] = 122.5
      return results
    return self.get_from_pocket('fake_labels', initializer=_fake_labels, local=True)

  def view(self):
    da = DaVinci()
    da.objects = [
      self.image,
      self.clipped_image,
      self.normalized_image,
      self.erode_dilate,
      self.connected_area,
      self.distance,
      self.watershed_label,
      self.segmentation_scheme,
      self.fake_labels
    ]
    da.object_titles = [
      'image',
      'clipped_images',
      'normalized_images',
      'erode_dilate',
      'connected_area',
      'distance,'
      'watershed_label',
      'segmentation_scheme'
    ]
    da.add_plotter(lambda x:da.imshow_pro(x, cmap=plt.jet()))
    da.show()

  # def segmented_cells(self):
  #   def _segmented_cells():
  #     image = self.image
  #     for bbox in self.bounding_boxes:




if __name__ == '__main__':
  import scipy.io as IO
  phase = IO.loadmat(r'D:\live_dead\naclo 05 save\1\green.mat')['green_matched']
  # phase = IO.loadmat(r'I:\live_dead\pbs save\1\phase.mat')['phase_matched']
  print(np.max(phase))
  seg = Segmenter(phase, (40, 140),erode_num=0, dilate_num=0, min_size=100, max_size=99999999)
  seg.view()


