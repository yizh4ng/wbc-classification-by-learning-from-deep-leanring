from lambo.zebravo.analyzer.analyzer import Analyzer
from lambo.analyzer.cropper import segmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lambo.zebravo.io_utili.saver import Saver



class Segmenter(Analyzer):
  def __init__(self, seg_threshold=None, save_size=150, min_size=10, max_size=90000):
    super(Segmenter, self).__init__()
    if seg_threshold is None:
      seg_threshold = [1, 5]
    self.seg_threshold = seg_threshold
    self.save_size = save_size
    self.min_size = min_size
    self.max_size = max_size


  def analyze(self, davincv, **kwargs):
    if davincv.objects[1] is None:
      print('No phase, please add a decoder')
      return None
    img = np.copy(davincv.objects[1])
    result = segmentation(img, seg_threshold=self.seg_threshold, save_size=self.save_size,
                          min_size=self.min_size, max_size=self.max_size)
    result = np.array(result).astype(int)
    return img, result

  def save(self, result, analyzer_save_path):
    img, result = result[0], result[1]
    for i, square in enumerate(result):
      single_cell = img[square[0][0]:square[1][0], square[0][1]:square[1][1]]
      Saver.save_image(analyzer_save_path, f'{self.frames}-{i}',
                       single_cell, 'png-npy')

  def visualize(self, result):
    if result is None:
      return None
    img, result = result[0], result[1]
    for i, square in enumerate(result):
      img = cv2.rectangle(img, (square[0][1], square[0][0]),
                          (square[1][1], square[1][0]), 6, 10)
    return img
