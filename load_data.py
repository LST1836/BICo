import pandas as pd
import numpy as np


col_r = ['R1-1-1', 'R2-1-2', 'R3-2-1', 'R4-2-2', 'R5-3-1', 'R6-3-2',
         'R7-3-3', 'R8-4-1', 'R9-4-2', 'R10-5-1', 'R11-5-2', 'R12-5-3']
col_i = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12']


def load_relevance_data(data_path):
    df = pd.read_csv(data_path)
    texts = list(df['tweet'])

    labels = np.array(df[col_r])

    return texts, labels


def load_ideology_data(data_path):
    df = pd.read_csv(data_path)
    texts = np.array(df['tweet'])

    related_mask = np.array(df[col_r]).transpose()
    i_label = np.array(df[col_i]).transpose()

    texts_input, i_label_input, facet_idx_input = [], [], []
    for i in range(12):
        mask_i = related_mask[i] == 1
        related_num = np.sum(mask_i)
        texts_input += list(texts[mask_i])
        i_label_input += list(i_label[i][mask_i])
        facet_idx_input += [i] * related_num

    i_label_input = np.array(i_label_input)
    i_label_input[i_label_input == 1] = 0
    i_label_input[i_label_input == 2] = 1
    i_label_input[(i_label_input == 3) | (i_label_input == 4)] = 2

    return texts_input, i_label_input, facet_idx_input


