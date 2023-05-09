import cv2
import numpy as np
import os

from roma import console
from ditto import VideoGenerator, VideoConfig, ImageConfig
import time
import datetime
from readerwriterlock import rwlock
from lambo.data_obj.cu_interferogram import Interferogram

marker = rwlock.RWLockFair()


class Fetcher():

  def __init__(self, fps=100, max_len=20, L=None, radius=80, invert=False):
    self.fps = fps
    self.radius = radius
    self.interferograms = None
    self.index = 0
    self.L = L
    self.max_len = max_len
    self.buffer = [None]
    self.save = []
    self.save_length=100
    self.save_path = ''
    self.peak_index = None
    self.invert = invert


  def _init(self):
    pass

  def _loop(self):
    raise NotImplementedError

  def append_to_buffer(self, data):
    self.buffer.append(data)
    if len(self.buffer) > self.max_len: self.buffer.pop(0)

  def fetch(self, davincv):
    self._init()
    frame = 0
    start_time = time.time()
    while davincv.play:
      if davincv.pause:
        time.sleep(0)
        continue
      self._loop()
      # if (davincv.decoder_cursor == 0 and davincv.analyzer_cursor == 0) or\
      # if  (davincv.object_cursor == 0):
      #   davincv.objects[davincv.object_cursor] = self.buffer[-1]
      davincv.objects[0] = self.buffer[-1]

      # deal with recording
      if davincv.recording[davincv.raw_channel]:
        self.save_path = davincv.save_path
        ig = Interferogram(self.buffer[-1], radius=self.radius, peak_index=davincv.background_ig.peak_index, invert=self.invert)
        self.save.append(ig.cropped_spectrum.get())
        ig.release()
        # handle memory
        if len(self.save) > self.save_length:
          np.savez( f'{self.save_path}' +'/'+ f'{datetime.datetime.now().strftime("%H-%M-%S-.%f")}.npz', np.array(self.save))
          self.save = []
      else:
        if len(self.save) > 0:
          np.savez(
            f'{self.save_path}' + '/' + f'{datetime.datetime.now().strftime("%H-%M-%S-.%f")}.npz',
            np.array(self.save))
          self.save = []

      frame += 1

      #Get fps
      if time.time() - start_time > 1:
        start_time = time.time()
        davincv.fetcher_fps = frame
        frame = 0

