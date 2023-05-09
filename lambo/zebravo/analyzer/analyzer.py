import time

class Analyzer():
  def __init__(self):
    self.fps_step = 0
    self.frames = 0
    self.tic = time.time()
    self.fps = 0
    self.thread_id = 999
    self.channel_num = 2

  def analyze(self,davincv):
    raise NotImplemented

  def visualize(self, result):
    raise NotImplemented

  def save(self, result, path):
    raise NotImplemented

  def get_fps(self):
    if time.time() - self.tic > 1:
      self.fps = self.fps_step
      self.fps_step = 0
      self.tic = time.time()
    return self.fps

  def loop(self, davincv):
    while davincv.play:
      if davincv.pause:
        time.sleep(0)
        continue

      if davincv.cursor[self.channel_num] == self.thread_id:
        result = self.analyze(davincv)
        if davincv.recording[self.channel_num] or davincv.snap_shot[self.channel_num]:
          self.save(result, davincv.channel_save_path[self.channel_num])
          davincv.snap_shot[self.channel_num] = False
        davincv.objects[self.channel_num] = self.visualize(result)
        davincv.decoder_fps = self.get_fps()
        self.fps_step += 1
        self.frames += 1
      time.sleep(0)
