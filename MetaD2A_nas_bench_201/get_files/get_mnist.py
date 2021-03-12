"""
@author: Hayeon Lee
2020/02/19
Script for downloading, and reorganizing aircraft 
for few shot classification
Run this file as follows:
    python get_data.py
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
import requests
import tarfile
from PIL import Image
import glob
import shutil
import pickle
import collections


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

dir_path = 'data/raw-data/mnist'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)
file_name = os.path.join(dir_path, 'mnist.tar.gz')

if not os.path.exists(file_name):
    print(f"Downloading {file_name}\n")
    download_file(
        'https://www.dropbox.com/s/fdx2ws1zh5rc9ae/mnist.tar.gz?dl=1',
        file_name)
    print("\nDownloading done.\n")
else:
    print(f"{file_name} has already been downloaded. Did not download twice.\n")

untar_folder = os.path.join(dir_path, "MNIST")
if not os.path.exists(untar_folder):
    tarname = file_name
    print("Untarring: {}".format(tarname))
    tar = tarfile.open(tarname)
    tar.extractall(dir_path)
    tar.close()
else:
    print(f"{untar_folder} folder already exists. Did not untarring twice\n")
os.remove(file_name)
