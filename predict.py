"""
Filename: predict.py
Author: yellower
"""
from torch.utils.data import DataLoader
import pandas as pd
from keras.models import load_model
import numpy as np
from keras import backend as K
from tqdm import tqdm
from utils import to_categorical,caculate_auc
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers


from keras.models import Model
from keras.layers import *

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    amino_acid_dict = {'*': 0, 'S': 1, 'L': 2, 'R': 3, 'D': 4, 'F': 5, 'Q': 6, 'V': 7, 'T': 8, 'A': 9, 'X': 10, 'N': 11,
                       'E': 12, 'W': 13, 'I': 14, 'P': 15, 'M': 16, 'Y': 17, 'K': 18, 'G': 19, 'H': 20, 'U': 21,
                       'C': 22}
    indentified_CPLM_all_K_concat = pd.read_table('path_for_your_sites.txt', sep='\t', index_col=0)
    model = load_model('path_for_your_model')
    fasta_K_emb_list = []
    for item in indentified_CPLM_all_K_concat['fasta_K']:
        fasta_K_emb_list.append([amino_acid_dict[i] for i in item])
        if len(item) != 61:
            print(len(item))
    fasta_K_emb_arrays = np.array(fasta_K_emb_list)
    score = model.predict(fasta_K_emb_arrays)
    print(score)

