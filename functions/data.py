from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import cv2
import numpy as np
from scipy import ndimage


def load_data(directory):
    x = []
    y = []
    fnames = listdir(directory + '/x/')
    for i in tqdm(range(len(fnames))):
        f = fnames[i]
        x.append(cv2.imread(join(directory + '/x/' , f)))
        y.append(ndimage.imread(join(directory + '/y/', f)))
    return np.array(x), np.expand_dims(np.array(y), axis = 3)
