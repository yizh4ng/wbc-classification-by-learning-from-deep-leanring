import cv2
import threading
import numpy as np
import time

class DaVincv():
  def __init__(self, title=None):
    self.objects = []
    self.titles = []
    self.layer_plotters = []
    self.object_cursor = 0
    self.layer_cursor = 0
    self.fps = 0
    self.fetcher_fps = 0
    self.title = title
    self.play = True

    self.keyboard_event ={
      'h': lambda: self.move_cursor(-1, 0),
      'j': lambda: self.move_cursor(0, 1),
      'k': lambda: self.move_cursor(0, -1),
      'l': lambda: self.move_cursor(1, 0),
      chr(27): lambda: self.stop()
    }

  def stop(self):
    print('Closing windows...')
    cv2.destroyAllWindows()
    self.play = False


  def add_plotter(self, plotter):
    self.layer_plotters.append(plotter)

  def move_cursor(self, x, y):
    self.object_cursor = (self.object_cursor + x) % len(self.objects)
    self.layer_cursor = (self.layer_cursor + y) % len(self.layer_plotters)

  def win_title(self):
    result = ''
    if len(self.objects) > 1:
      result = '[{}/{}]'.format(self.object_cursor + 1, len(self.objects))
    if len(self.layer_plotters) > 1:
      result += '[{}/{}]'.format(
        self.layer_cursor + 1, len(self.layer_plotters))

    if self.title is not None: result = ' '.join([result, self.title])
    if not result: return ''
    return result

  def show(self):
    while self.play:
      k = self.layer_plotters[self.layer_cursor](self.objects[self.object_cursor])
      if k == -1:
        # no keyboard input
        continue
      else:
        # process keyboard input
        if chr(k) not in self.keyboard_event.keys():
          print(f'Key {chr(k)} not registered.')
        else:
          self.keyboard_event[chr(k)]()

  def imshow(self, x, color_bar=cv2.COLORMAP_JET):
    cv2.setWindowTitle('main', self.win_title())
    x -= np.min(x)
    x = cv2.applyColorMap((x / np.max(x) * 128).astype(np.uint8), color_bar)
    cv2.imshow('main', x)
    k = cv2.waitKey(0)
    return k



if __name__ == '__main__':
  da = DaVincv()
  da.objects = [np.ones([2000, 1000])]
  da.add_plotter(da.imshow)
  da.show()