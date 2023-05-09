from typing import List
import os

import cv2

from roma import console
from lambo.misc.local import walk
from lambo.misc.io_utils import save, load
from lambo.data_obj.interferogram import Interferogram



class Retriever():
  def __init__(self, interferograms):
    self.interferograms = interferograms

  @classmethod
  def get_save_file_name(cls, radius, boost, save_keys):
    file_name = 'ig-rad{}-boost-{}'.format(radius, boost)
    if save_keys is not None:
      for save_key in save_keys:
        file_name += '-{}'.format(save_key)
    file_name +='.save'
    return file_name

  @classmethod
  def read_interferograms(cls, path:str, radius=80, boost=False,
                          save_keys=None):
    save_fn = cls.get_save_file_name(radius, boost, save_keys)
    save_path = os.path.join(path, save_fn)
    if radius is not None and os.path.exists(save_path):
      console.show_status('Loading `{}` ...'.format(save_path))
      return load(save_path)

    # Find the file organization type by looking at the sub-folders
    interferograms = []
    subfolders = walk(path, type_filter='folder', return_basename=True)
    # Remove folders which should be ignored
    if 'trash' in subfolders: subfolders.remove('trash')

    if 'sample' in subfolders and 'bg' in subfolders:
      # Case (1)
      console.show_status('Reading files organized as `sample/bg` ...')
      sample_folder, bg_folder = [
        os.path.join(path, fn) for fn in ('sample', 'bg')]
      # Scan all sample files in sample folder
      for sample_path in walk(
          sample_folder, type_filter='file', pattern='*.tif*'):
        fn = os.path.basename(sample_path)
        bg_path = os.path.join(bg_folder, fn)
        if not os.path.exists(bg_path):
          # console.warning(
          #   ' ! Background file `{}` does not exist'.format(bg_path))
          bg_path = walk(bg_folder, type_filter='file', pattern='*.tif*')[0]
          ig = Interferogram.imread(sample_path, bg_path, radius, boost=boost)
          interferograms.append(ig)
        else:
          ig = Interferogram.imread(sample_path, bg_path, radius, boost=boost)
          interferograms.append(ig)
    elif len(subfolders) == 0 and all(['bg' not in path for path in
                                       walk(
                                         path, type_filter='file',
                                         pattern='*.tif*')
                                       ]):
      # Case (2)
      console.show_status('Reading files organized as `odd/even` for {}'.format(path))
      file_list = walk(
        path, type_filter='file', pattern='*.tif*', return_basename=True)
      while len(file_list) > 0:
        fn = file_list.pop(0)
        # Get int id
        index = int(fn.split('.')[0])
        # If file is sample
        if index % 2 == 1:
          sample_fn = fn
          bg_fn = '{}.tif'.format(index + 1)
          mate = bg_fn
        else:
          bg_fn = fn
          sample_fn = '{}.tif'.format(index - 1)
          mate = sample_fn
        # Check if mate exists
        if mate not in file_list:
          console.warning(' ! Mate file `{}` of `{}` does not exist'.format(
            mate, fn))
          continue
        # Remove mate from list and append sample/bg path to pairs
        file_list.remove(mate)
        # Read interferogram
        ig = Interferogram.imread(
          *[os.path.join(path, f) for f in (sample_fn, bg_fn)], radius, boost=boost)
        if ig is None: continue
        interferograms.append(ig)
    else:
      # Case (3)
      console.show_status(
        'Reading files from `{}` organized as video sequences ...'.format(
          path))

      # Read background
      bg = Interferogram(cv2.imread(os.path.join(path, 'bg.tif'), 0),
                         radius=radius, boost=boost)
      paths = walk(path, 'file', pattern='*.tif')
      for i, p in enumerate(paths):
        console.print_progress(i, len(paths))
        if 'bg' in p: continue
        if any([v in p for v in ['red', 'green', 'fluo']]): continue
        ig = Interferogram(cv2.imread(p, 0), radius=radius, boost=boost)
        # Using peak_index from background
        ig.peak_index = bg.peak_index
        # Set the same background to all interferograms
        ig.bg_ig = bg
        interferograms.append(ig)

    console.show_status('{} interferograms have been read.'.format(
      len(interferograms)))

    # Save if required
    if radius is not None and save_keys is not None:
      Retriever.save(interferograms, save_path, keys=save_keys)

    return interferograms

  @classmethod
  def generate_interferograms(cls, path:str, radius=80, boost=True):
    # save_fn = cls.get_save_file_name(radius, boost, save_keys)
    # save_path = os.path.join(path, save_fn)
    # if radius is not None and os.path.exists(save_path):
    #   console.show_status('Loading `{}` ...'.format(save_path))
    #   return load(save_path)

    # Find the file organization type by looking at the sub-folders
    subfolders = walk(path, type_filter='folder', return_basename=True)
    # Remove folders which should be ignored
    if 'trash' in subfolders: subfolders.remove('trash')

    if 'sample' in subfolders and 'bg' in subfolders:
      # Case (1)
      console.show_status('Reading files organized as `sample/bg` ...')
      sample_folder, bg_folder = [
        os.path.join(path, fn) for fn in ('sample', 'bg')]
      # Scan all sample files in sample folder
      for sample_path in walk(
          sample_folder, type_filter='file', pattern='*.tif*'):
        fn = os.path.basename(sample_path)
        bg_path = os.path.join(bg_folder, fn)
        if not os.path.exists(bg_path):
          # console.warning(
          #   ' ! Background file `{}` does not exist'.format(bg_path))
          bg_path = walk(bg_folder, type_filter='file', pattern='*.tif*')[0]
          ig = Interferogram.imread(sample_path, bg_path, radius, boost=boost)
          # interferograms.append(ig)
          yield ig
        else:
          ig = Interferogram.imread(sample_path, bg_path, radius, boost=boost)
          # interferograms.append(ig)
          yield ig
    elif len(subfolders) == 0:
      # Case (2)
      console.show_status('Reading files organized as `odd/even` for {}'.format(path))
      file_list = walk(
        path, type_filter='file', pattern='*.tif*', return_basename=True)
      while len(file_list) > 0:
        fn = file_list.pop(0)
        # Get int id
        index = int(fn.split('.')[0])
        # If file is sample
        if index % 2 == 1:
          sample_fn = fn
          bg_fn = '{}.tif'.format(index + 1)
          mate = bg_fn
        else:
          bg_fn = fn
          sample_fn = '{}.tif'.format(index - 1)
          mate = sample_fn
        # Check if mate exists
        if mate not in file_list:
          console.warning(' ! Mate file `{}` of `{}` does not exist'.format(
            mate, fn))
          continue
        # Remove mate from list and append sample/bg path to pairs
        file_list.remove(mate)
        # Read interferogram
        ig = Interferogram.imread(
          *[os.path.join(path, f) for f in (sample_fn, bg_fn)], radius, boost=boost)
        if ig is None: continue
        # interferograms.append(ig)
        yield ig
    else:
      # Case (3)
      console.show_status(
        'Reading files from `{}` organized as video sequences ...'.format(
          path))

      # Read background
      bg = Interferogram.imread(os.path.join(path, 'bg.tif'), radius=radius,
                                boost=boost)
      paths = walk(path, 'file', pattern='*.tif')
      for i, p in enumerate(paths):
        console.print_progress(i, len(paths))
        if bg in p: continue
        ig = Interferogram.imread(p, radius=radius, boost=boost)
        # Using peak_index from background
        ig.peak_index = bg.peak_index
        # Set the same background to all interferograms
        ig.bg_ig = bg
        # interferograms.append(ig)
        yield ig

    # console.show_status('{} interferograms have been read.'.format(
    #   len(interferograms)))

    # Save if required
    # if radius is not None and save_keys is not None:
    #   Retriever.save(interferograms, save_path, keys=save_keys)

    # return interferograms


  @staticmethod
  def save(interferograms: List[Interferogram], save_path,
           keys=('flattened_phase',)):
    total = len(interferograms)
    console.show_status('Generating {} ...'.format(', '.join(
      ['`{}`'.format(k) for k in keys])))
    for i, ig in enumerate(interferograms):
      console.print_progress(i, total)
      for key in keys:
        _ = getattr(ig, key)
        # The data can only be saved when they are localized
        # We only save the following intermediate results to save memory
        ig.localize(key)
        ig.bg_ig.release()
        ig.release()
    save(interferograms, save_path)
    console.show_status('Interferograms saved to `{}`'.format(save_path))

if __name__ == '__main__':
  from lambo.data_obj.visualizers.interferogram_visualizer\
    import InterferogramVisualizer

  path = None
  # path = r'G:\projects\data_old\05-bead'
  if path is None:
    from tkinter import filedialog
    path = filedialog.askdirectory()

  # path = r'G:\projects\data_old\05-bead'
  inteferograms = Retriever.read_interferograms(path, radius=80, boost=True,
                                                save_keys=('flattened_phase',))



  InterferogramVisualizer().visualize(inteferograms)
  Phase = inteferograms[0].flattened_phase
  for i,_ in enumerate(inteferograms):
    Phase = inteferograms[i].flattened_phase

  # Retriever,save()



