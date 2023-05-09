import os
import time

import numpy as np

from roma import console

from lambo.data_obj.interferogram import Interferogram
from lambo.analyzer.retriever import Retriever
from lambo.zebra.io.inflow import Inflow
from ditto import VideoGenerator, VideoConfig, ImageConfig


class DittoFetcher(Inflow):

  def __init__(self, fps=10, max_len=20, L=None):
    super(DittoFetcher, self).__init__(max_len)
    self.fps = fps
    self.interferograms = None
    self.index = 0
    self.L = L


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
    time.sleep(1 / self.fps)
    self.index += 1



