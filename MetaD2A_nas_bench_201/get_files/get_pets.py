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

dir_path = 'data/raw-data/pets'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

full_name = os.path.join(dir_path, 'test15.pth')
if not os.path.exists(full_name):
  print(f"Downloading {full_name}\n")
  download_file('https://www.dropbox.com/s/kzmrwyyk5iaugv0/test15.pth?dl=1', full_name)
  print("Downloading done.\n")
else:
  print(f"{full_name} has already been downloaded. Did not download twice.\n")

full_name = os.path.join(dir_path, 'train85.pth')
if not os.path.exists(full_name):
  print(f"Downloading {full_name}\n")
  download_file('https://www.dropbox.com/s/w7mikpztkamnw9s/train85.pth?dl=1', full_name)
  print("Downloading done.\n")
else:
  print(f"{full_name} has already been downloaded. Did not download twice.\n")
