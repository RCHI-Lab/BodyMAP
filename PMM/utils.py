import pickle as pkl
import numpy as np


def load_pickle(filename):
    try:
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding = 'latin1')
    except:
        print (f'Error in {filename}')
        return None


def load_data_lines(data_lines_file):
    with open(data_lines_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def concatenate(trg, src):
    return np.concatenate((trg, src)) if trg is not None else src

