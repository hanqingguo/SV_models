#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp

class SpeakerDatasetTIMITPreprocessed(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #########################################
        # Update Jun 21, produce label as spk id
        #########################################
        # 每次拿到一个人的几段音频
        np_file_list = os.listdir(self.path)

        if self.shuffle:
            selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
        else:
            selected_file = np_file_list[idx]
        # print(self.path)
        label = int(selected_file.split(".")[0][7:])
        label_per_spk = np.repeat(label, hp.train.M)

        utters = np.load(os.path.join(self.path, selected_file))  # load utterance spectrogram of selected speaker
        if utters.shape[0] > 0:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)  # select M utterances per speaker
            utterance = utters[utter_index]
            # print(utterance.shape, selected_file)
            utterance = utterance[:, :, :160]
            utterance = torch.tensor(np.transpose(utterance, axes=(0, 2, 1)))  # transpose [batch, frames, n_mels]
            # [6, 160, 40] 6=> utterance per speaker, 160 frames, 40 mel
            return utterance, label_per_spk
        else:
            print("Find less utters, restart select speaker")
            return self.__getitem__(idx)
