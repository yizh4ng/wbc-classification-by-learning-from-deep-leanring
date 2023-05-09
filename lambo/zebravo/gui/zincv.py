import cv2
import os
import threading
import numpy as np
from readerwriterlock import rwlock
import time
import datetime
from lambo.gui.vincv.davincv import DaVincv
from lambo.data_obj.cu_interferogram import Interferogram
import matplotlib.pyplot as plt
from lambo.zebravo.io_utili.saver import Saver

marker = rwlock.RWLockFair()

class Zincv(DaVincv):
  def __init__(self, fps = 100, title=None, invert=False):
    super(Zincv, self).__init__(title)
    self.invert = invert
    self.fps = 0
    self.fetcher = None
    self.fetcher_fps = 0
    self.decoder_fps =0
    self.analyzers = []
    self.decoders = []
    self.objects = [None, None, None]
    self.pause = True
    self.keyboard_event.update({
      ' ': lambda: self.pause_or_play(),
    })
    self.add_plotter(self.imshow)
    self.background = None
    self.background_ig = None
    self.save_path = ''
    self.phase_count = 0
    self.raw_count = 0
    self.gui = None
    self.snapshot_analysis = False
    self.recording = [False, False, False]
    self.snap_shot = [False, False, False]
    self.raw_channel = 0
    self.decoder_channel = 1
    self.analyzer_channel = 2
    self.channel_save_path = ['', '', '']
    self.cursor = [0, 0, 0 ]

  def change_channel(self):
    self.object_cursor = (self.object_cursor + 1) % len(self.objects)


  def change_decoder(self):
    self.cursor[self.decoder_channel] =(self.cursor[self.decoder_channel] + 1)%(len(self.decoders) + 1)

  def change_analyzer(self):
    self.cursor[self.analyzer_channel] = (self.cursor[self.analyzer_channel] + 1)%(len(self.analyzers) + 1)

  def pause_or_play(self):
    self.pause = not self.pause
    print(f'Player: {not bool(self.pause)}')

  def change_recording(self):
    if self.background is None:
      print('a background must be set before recording.')
      return
    if not self.recording[self.raw_channel]:
      self.save_path = './sample/' + str(datetime.datetime.now().strftime("%H-%M-%S-.%f"))
      Saver.save_image(self.save_path + '/background', 'background',
                       self.background_ig.cropped_spectrum, fmts='npz')
    self.recording[self.raw_channel] = ~self.recording[self.raw_channel]
    print(f'recording {bool(self.recording[self.raw_channel])}')

  def change_analyzer_recording(self):
    if not self.recording[self.analyzer_channel]:
      if not os.path.exists('./analyzed_results'):
        os.mkdir('./analyzed_results')
      self.channel_save_path[self.analyzer_channel] = './analyzed_results/' + str(datetime.datetime.now().strftime("%H-%M-%S-.%f"))
      os.mkdir(self.channel_save_path[self.analyzer_channel])
    self.recording[self.analyzer_channel] = ~self.recording[self.analyzer_channel]
    print(f'analyzer recording {bool(self.recording[self.analyzer_channel])}')

  def capture_analysis(self):
    self.snap_shot[self.analyzer_channel] = True
    self.channel_save_path[self.analyzer_channel] = './analyzed_results/single_frame/' + str(
      datetime.datetime.now().strftime("%H-%M-%S-.%f"))
    if ~os.path.exists(self.channel_save_path[self.analyzer_channel]):
      os.makedirs(self.channel_save_path[self.analyzer_channel])

  def change_decoder_recording(self):
    if not self.recording[self.decoder_channel]:
      if not os.path.exists('./analyzed_results'):
        os.mkdir('./analyzed_results')
      self.channel_save_path[self.decoder_channel] = './captured_phase/' + str(datetime.datetime.now().strftime("%H-%M-%S-.%f"))
      os.mkdir(self.channel_save_path[self.decoder_channel])
    self.recording[self.decoder_channel] = ~self.recording[self.decoder_channel]
    print(f'analyzer recording {bool(self.recording[self.decoder_channel])}')

  def capture_phase(self):
    self.snap_shot[self.decoder_channel] = True
    self.channel_save_path[self.decoder_channel] = f'./captured_phase/single_frame/{str(datetime.datetime.now().strftime("%H-%M-%S-.%f"))}'
    if ~os.path.exists(self.channel_save_path[self.decoder_channel]):
      os.makedirs(self.channel_save_path[self.decoder_channel])


  def capture_raw(self):
    root_path = f'./captured_raw/{str(datetime.datetime.now().strftime("%H-%M-%S-.%f"))}'
    Saver.save_image(root_path, self.raw_count, self.objects[self.raw_channel],
                     fmts='npy-png')
    print(f"Raw interferograms saved to {root_path + f'/{self.raw_count}.npy'}")
    self.raw_count += 1

  def set_background(self):
    read_marker = marker.gen_rlock()
    read_marker.acquire()
    self.background = self.objects[self.raw_channel]
    self.background_ig = Interferogram(self.background, radius=self.fetcher.radius, invert=self.invert)
    print('Background has been set.')
    read_marker.release()

  def set_fetcher(self, fetcher):
    self.fetcher = fetcher

  def add_decoder(self, decoder):
    self.decoders.append(decoder)

  def add_analyzer(self, analyzer):
    self.analyzers.append(analyzer)

  def _loop(self):
    frame = 0
    start_time = time.time()
    read_marker = marker.gen_rlock()
    while self.play:
      read_marker.acquire()
      k = self.layer_plotters[self.layer_cursor](self.objects[self.object_cursor])
      read_marker.release()
      frame += 1
      if time.time() - start_time > 1:
        start_time = time.time()
        self.fps = frame
        frame = 0
      if k == -1:
        # no keyboard input
        continue
      else:
        # process keyboard input
        if chr(k) not in self.keyboard_event.keys():
          print(f'Key {chr(k)} not registered.')
        else:
          self.keyboard_event[chr(k)]()
    time.sleep(0)

  def imshow(self, x, color_bar=cv2.COLORMAP_JET):
    if x is None:
      img = cv2.putText(np.ones((500, 500, 3)), 'opening gates...', (70, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
      cv2.imshow('main', img)
      k = cv2.waitKey(1)
      return k
    cv2.setWindowTitle('main', self.win_title() + f'Player fps = '
                       + str(self.fps) + f' Fetcher fps = {self.fetcher_fps}'
                                      + f' Decoder fps = {self.decoder_fps}')
    x = np.copy(x)
    if any(np.array(x.shape) > 1000):
      w, h = x.shape[0],x.shape[1]
      x = cv2.resize(x, (int(0.3 * w), int(0.3 * w)))
    # y = np.median(x)
    # x -= np.mean(x).astype(np.uint8)
    x -= np.min(x)
    if self.cursor == 1:
      x -= np.percentile(x, 5).astype(np.uint8)
      x[x < 0] = 0
    x = cv2.applyColorMap((x / np.max(x) * 256).astype(np.uint8), color_bar)
    cv2.imshow('main', x)
    k = cv2.waitKey(1)
    return k

  def show_text(self, text):
    img = cv2.putText(np.ones((500, 500, 3)), text, (70, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Welcome', img)
    k = cv2.waitKey(0)
    while chr(k) != ' ':
      k = cv2.waitKey(0)
    self.keyboard_event[chr(k)]()
    cv2.destroyWindow('Welcome')

  def display(self):
    self.show_text('press space to start')
    p1 = threading.Thread(target=self._loop)
    p2 = threading.Thread(target=self.fetcher.fetch, args=[self])
    p = []
    for i, decoder in enumerate(self.decoders):
      decoder.thread_id = i + 1
      p.append(threading.Thread(target=decoder.loop, args=[self]))
    for i, analyzer in enumerate(self.analyzers):
      analyzer.thread_id = i + 1
      p.append(threading.Thread(target=analyzer.loop, args=[self]))
    if not self.pause:
      p2.start()
      p1.start()
      for pi in p:
        pi.start()
    if self.gui is not None:
      gui_thread = threading.Thread(target=self.gui._loop, args=[self])
      gui_thread.start()

if __name__ == '__main__':
  da = Zincv()
  da.display()