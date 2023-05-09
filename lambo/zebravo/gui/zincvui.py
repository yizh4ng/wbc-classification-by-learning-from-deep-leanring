import threading
import time

import numpy as np
import cv2
import cvui
from lambo.zebravo.gui.zincv import Zincv


class Zincvui():
  def __init__(self, zincv:Zincv):
    self.win_title = 'Control Panel'
    self.board = np.zeros([300, 500], np.uint8)
    self.zincv = zincv
    self.zincv.gui = self
    # cv2.setMouseCallback(self.win_title,mouse_event)
    self.start = [False]

  def _loop(self, zincv):
    cvui.init(self.win_title)
    while zincv.play:
      self.board[:] = 49
      if cvui.button(self.board, 10, 15, 'Capture Raw'):
        zincv.capture_raw()
      if cvui.button(self.board, 10, 50, 'Capture Phase'):
        zincv.capture_phase()
      if cvui.button(self.board, 10, 85, 'Capture Analysis'):
        zincv.capture_analysis()
      if cvui.button(self.board, 10, 120, 'Set Background'):
        zincv.set_background()
      if cvui.button(self.board, 10, 155, 'Change Channel'):
        zincv.change_channel()
      if cvui.button(self.board, 10, 190, 'Change Decoder'):
        zincv.change_decoder()
      if cvui.button(self.board, 10, 225, 'Change Analyzer'):
        zincv.change_analyzer()
      if zincv.pause:
        if cvui.button(self.board, 10, 260, 'Play'):
          zincv.pause_or_play()
      else:
        if cvui.button(self.board, 10, 260, 'Pause'):
          zincv.pause_or_play()

      if not self.zincv.recording[self.zincv.raw_channel]:
        if cvui.button(self.board, 200, 15, 'Start Recording Raw Data'):
          zincv.change_recording()
      else:
        if cvui.button(self.board, 200, 15, 'Stop Recording Raw Data'):
          zincv.change_recording()

      if not self.zincv.recording[self.zincv.decoder_channel]:
        if cvui.button(self.board, 200, 50, 'Start Recording Phase'):
          zincv.change_decoder_recording()
      else:
        if cvui.button(self.board, 200, 50, 'Stop Recording Phase'):
          zincv.change_decoder_recording()

      if not self.zincv.recording[self.zincv.analyzer_channel]:
        if cvui.button(self.board, 200, 85, 'Start Recording Analyzer'):
          zincv.change_analyzer_recording()
      else:
        if cvui.button(self.board, 200, 85, 'Stop Recording Analyzer'):
          zincv.change_analyzer_recording()

      if cvui.button(self.board, 200, 225, 'Quit'):
        zincv.play = False
      cvui.update(self.win_title)
      cv2.imshow(self.win_title, self.board)
      #
      cv2.waitKey(1)


  def display(self):
    p = threading.Thread(target=self._loop)
    p.start()

if __name__ == '__main__':
  # import numpy as np
  # import cv2
  # import cvui
  # print(cv2.__version__)
  # WINDOW_NAME = 'CVUI Test'
  # cvui.init(WINDOW_NAME)
  # frame = np.zeros((200, 400), np.uint8)
  #
  # # use an array/list because this variable will be changed by cvui
  # checkboxState = [False]
  #
  # while True:
  #   frame[:] = 49
  #
  #   # Render the checkbox. Notice that checkboxState is used AS IS,
  #   # e.g. simply "checkboxState" instead of "checkboxState[0]".
  #   # Only internally that cvui will use checkboxState[0].
  #   cvui.checkbox(frame, 10, 15, 'My checkbox', checkboxState)
  #
  #   # Check the state of the checkbox. Here you need to remember to
  #   # use the first position of the array/list because that's the
  #   # one being used by all cvui components that perform changes
  #   # to external variables.
  #   if checkboxState[0] == True:
  #     print('Checkbox is checked')
  #   cv2.imshow('CVUI Test', frame)
  #   if cv2.waitKey(20) == 27:
  #     break
  zv =  Zincvui(None)
  zv.display()