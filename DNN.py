"""
Filename: DNN.py
Author: yellower
"""

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
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import *



def DNN (indentified_CPLM_all_K_concat,maxlen,hidden_size1,hidden_size2,dropout_size):
    amino_acid_dict = {'*': 0, 'S': 1, 'L': 2, 'R': 3, 'D': 4, 'F': 5, 'Q': 6, 'V': 7, 'T': 8, 'A': 9, 'X': 10, 'N': 11,
                       'E': 12, 'W': 13, 'I': 14, 'P': 15, 'M': 16, 'Y': 17, 'K': 18, 'G': 19, 'H': 20, 'U': 21,
                       'C': 22}

    fasta_K_emb_list = []
    for item in indentified_CPLM_all_K_concat['fasta_K']:
        fasta_K_emb_list.append([amino_acid_dict[i] for i in item])
        if len(item)!=61:
            print(len(item))


    labels_arrays = np.array(indentified_CPLM_all_K_concat['label'])
    fasta_K_emb_arrays = np.array(fasta_K_emb_list)
    opt = Config()
    skf = StratifiedKFold(n_splits=opt.num_folds, shuffle=True)
    logits_test_concat = np.array([])
    labels_test_concat = np.array([])
    inputs = layers.Input(shape=(maxlen,))

    O_seq = Dense(hidden_size1, activation='selu')(inputs)
    O_seq = Dropout(dropout_size)(O_seq)
    O_seq = Dense(hidden_size2, activation='selu')(O_seq)
    O_seq = Dropout(dropout_size)(O_seq)
    outputs = Dense(2, activation='softmax')(O_seq)

    for fold, (train_idx, val_idx) in enumerate(skf.split(fasta_K_emb_arrays, labels_arrays)):
        fasta_K_emb_train, labels_train, fasta_K_emb_test, labels_test = fasta_K_emb_arrays[train_idx], labels_arrays[train_idx], fasta_K_emb_arrays[val_idx], labels_arrays[val_idx]
        labels_train_cate = to_categorical(labels_train, 2)
        labels_test_cate = to_categorical(labels_test, 2)
        DNN_model = Model(inputs,outputs)
        DNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        class_weight = {0: 1 / len(fasta_K_emb_train), 1: 1 / len(fasta_K_emb_test)}
        history = transformer_model.fit(fasta_K_emb_train, labels_train_cate, epochs=opt.epochs,
                                       batch_size=opt.batch_size,
                                       validation_data=(fasta_K_emb_test, labels_test_cate),class_weight=class_weight)
        logits_test = transformer_model.predict(fasta_K_emb_test)
        fpr, tpr, roc_auc = caculate_auc(labels_test, logits_test[:, 1])
        print(fold)
        print(roc_auc)
        logits_test_concat = np.append(logits_test_concat, logits_test[:, 1])
        labels_test_concat = np.append(labels_test_concat, labels_test)




