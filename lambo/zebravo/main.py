from lambo.zebravo.io_utili.ditto_fetcher import DittoFetcher
from lambo.zebravo.io_utili.pseudo import PseudoFetcher
from lambo.zebravo.io_utili.hikv import HIkVFetcher
from lambo.zebravo.io_utili.flir import FLIRFetcher
from lambo.zebravo.gui.zincv import Zincv
from lambo.zebravo.gui.zincvui import Zincvui
from lambo.zebravo.decoder.substracter import Subtractor
from lambo.zebravo.decoder.differentiate_integrate import DI
# from lambo.zebravo.decoder.di import DI
from lambo.zebravo.decoder.DNN import DNN
from lambo.zebravo.analyzer.segmentation import Segmenter
import os



if __name__ == '__main__':
  da = Zincv(invert=False)
  trial_root = r'G:\projects\data_old'
  trial_names = ['01-3t3','05-bead','04-rbc', '03-Hek']
  path = os.path.join(trial_root, trial_names[1])
  da.set_fetcher(PseudoFetcher(path,  fps=100,  seq_id=4))
  # da.set_fetcher(DittoFetcher(max_len=20))
  # da.set_fetcher(FLIRFetcher(radius=80))
  # da.set_fetcher(HIkVFetcher())
  # da.add_decoder(Subtractor(radius=80,boost=False, invert=True))
  # da.add_decoder(Subtractor(radius=80,boost=True, invert=True))
  # da.add_decoder(Subtractor(radius=80,boost=False, invert=False))
  # da.add_decoder(Subtractor(radius=80,boost=True, invert=False))
  # da.add_decoder(DI(radius=80, boost=False))
  da.add_decoder(DNN())
  da.add_analyzer(Segmenter())
  da.gui = Zincvui(zincv=da)

  da.display()
  # gui.display()
