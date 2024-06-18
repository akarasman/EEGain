import os
import re
import mne
import torch
import pickle
import logging
import scipy.io
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from pymatreader import read_mat
from scipy.io import loadmat
from ..transforms import Construct
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple
from torch.utils.data import Dataset

from .datasets import EEGDataset

logger = logging.getLogger("Dataset_features")

class DREAMER_feat(EEGDataset):
    def __init__(self, root: str, label_type: str, ground_truth_threshold : float, transform=None, **kwargs):
        self.root = root
        data = read_mat(f'{root}/feat_array_all_DREAMER.mat')['feat_array_all_DREAMER'][1::2,:]
        colnames = read_mat(f'{root}/colnames_DREAMER.mat')['colnames']
        self.df = pd.DataFrame(data, columns=colnames,)
        self.label_type = label_type
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = None
        self.unique_identifier = 'DREAMER' 
        self.subject_ids = self.df['subj'].unique().astype(int)
        self.trials = self.df['task'].unique().astype(int)
        self.mapping_list = self._create_user_recording_mapping()

    def _create_user_recording_mapping(self) -> Dict[int, List[str]]:
        user_session_info: Dict[int, List[str]] = defaultdict(list)
        for subject_id in self.subject_ids:
            for trial in self.trials:
                if float(trial) not in self.df[self.df['subj'] == subject_id]['task'].values:
                    continue
                trial_name = f"{subject_id}_{self.unique_identifier}_{trial}"
                user_session_info[subject_id].append(trial_name)
            
            
        return user_session_info
    
    def __get_subject_ids__(self) -> List[int]:
        return self.subject_ids

    def __get_subject__(self, subject_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        
        label_array = {}
        data_array = {}
        sessions = self.mapping_list[subject_index]
        
        for session in sessions:
            subject_index = int(session.split(self.unique_identifier)[0][:-1])
            trial = int(session.split(self.unique_identifier)[1][1:])
            
            subject_data = self.df[self.df['subj'] == subject_index]
            row =  subject_data[subject_data['task'] == trial]
            label_type = 'valence' if self.label_type == 'V' else 'arousal'
            label = int(row[label_type] <= self.ground_truth_threshold)
            
            label_array[f'{int(subject_index)}/{int(trial)}'] = label 
            data_array[f'{int(subject_index)}/{int(trial)}'] = row.values[:,2:-2]
        
        # data_array = { key: value
        #                for key, value in data_array.items() }
        return data_array, label_array

    def __get_trials__(self, sessions, subject_ids):
        
        label_array = {}
        data_array = {}
        for session in sessions:
            subject_index = int(session.split(self.unique_identifier)[0][:-1])
            trial = int(session.split(self.unique_identifier)[1][1:])
            
            subject_data = self.df[self.df['subj'] == subject_index]
            row =  subject_data[subject_data['task'] == trial]
            label_type = 'valence' if self.label_type == 'V' else 'arousal'
            label = int(row[label_type] <= self.ground_truth_threshold)
            
            label_array[f'{int(subject_index)}/{int(trial)}'] = label 
            data_array[f'{int(subject_index)}/{int(trial)}'] = row.values[:,2:-2]
        
        # data_array = { key: value
        #                for key, value in data_array.items() }
        return data_array, label_array