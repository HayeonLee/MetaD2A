###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import os
from tqdm import tqdm
import requests
import zipfile

def download_file(url, filename):
  """
  Helper method handling downloading large files from `url`
  to `filename`. Returns a pointer to `filename`.
  """
  chunkSize = 1024
  r = requests.get(url, stream=True)
  with open(filename, 'wb') as f:
    pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
    for chunk in r.iter_content(chunk_size=chunkSize):
      if chunk: # filter out keep-alive new chunks
        pbar.update (len(chunk))
        f.write(chunk)
  return filename

file_name = 'ckpt_max_corr.pt'
dir_path = 'results/predictor/model'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)
file_name = os.path.join(dir_path, file_name)
if not os.path.exists(file_name):
  print(f"Downloading {file_name}\n")
  download_file('https://www.dropbox.com/s/1l73vq2orv0chso/ckpt_max_corr.pt?dl=1', file_name)
  print("Downloading done.\n")
else:
  print(f"{file_name} has already been downloaded. Did not download twice.\n")
