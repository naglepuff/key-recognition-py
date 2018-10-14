import numpy as np 

from scipy.io import wavfile 
import scipy.fftpack as fft 

import matplotlib.pyplot as plt 

import sys

DEFAULT_FILENAME = "i_iv_v_keyA.wav"
KEY_SIGNATURE_TEMPLATE = [0, 2, 4, 5, 7, 9, 11] 


def get_filename():
    """gets the filename as input. filename should be the name of a wav file"""
    try:
        filename = sys.argv[1] 
    except IndexError:
        filename = DEFAULT_FILENAME 

    return filename

def find_nearest_index(array, value):
    """Given an array and a value, find the index of the element of given array closest to given value"""
    array = np.asarray(array) 
    index = (np.abs(array - value)).argmin() 
    return index 

def make_key_signatures():
    keys = [] 
    for i in range(0, 12):
        keys += [[(j + i) % 12 for j in KEY_SIGNATURE_TEMPLATE]] 
    return keys