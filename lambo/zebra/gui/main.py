from lambo.zebra.io.ditto_fetcher import DittoFetcher
from lambo.zebra.decoder.decoder_nobg import Subtracter
from zinci import Zinci


z = Zinci(height=6, width=6, max_fps=50)
z.set_fetcher(DittoFetcher(fps=50, max_len=40))
z.set_decoder(Subtracter())
z.set_decoder(Subtracter(boosted=True))

z.display()