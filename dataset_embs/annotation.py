import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class Annotation_Dataset(Dataset):
    def __init__(self, data_path, split, audio_embs):
        self.data_path = data_path
        self.split = split
        self.audio_embs = audio_embs
        self.get_split()
        self.get_file_list()
        
    def get_split(self):
        track_split = json.load(open(os.path.join(self.data_path, "Annotation", "Annotation_split.json"), "r"))
        self.train_track = track_split['train_track']
        self.valid_track = track_split['valid_track']
        self.test_track = track_split['test_track']

    def get_file_list(self):
        self.list_of_label = json.load(open(os.path.join(self.data_path, "Annotation", "Annotation_tags.json"), 'r')) # 이미 소문자임
        self.tag_to_idx = {i:idx for idx, i in enumerate(self.list_of_label)}
        
        with open(os.path.join(self.data_path, "Annotation", "Annotation.json"), 'r') as file:
            annotation_data = json.load(file)
            annotation = {str(item['track_id']): item for item in annotation_data}
        
        if self.split == "TRAIN":
            self.fl = [annotation[str(i)] for i in self.train_track]
        elif self.split == "VALID":
            self.fl = [annotation[str(i)] for i in self.valid_track]
        elif self.split == "TEST":
            self.fl = [annotation[str(i)] for i in self.test_track]
        elif self.split == "ALL":
            self.fl = list(annotation.values())
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del annotation
        
    def tag_to_binary(self, text):
        binary = np.zeros([len(self.list_of_label),], dtype=np.float32)
        if isinstance(text, str):
            binary[self.tag_to_idx[text]] = 1.0
        elif isinstance(text, list):
            for tag in text:
                binary[self.tag_to_idx[tag]] = 1.0
        return binary

    def get_train_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        audio_tensor = self.audio_embs[str(item['track_id'])]
        return {
            "audio":audio_tensor, 
            "binary":binary
            }
        
    def get_eval_item(self, index):
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = ", ".join(tag_list)
        tags = self.list_of_label
        track_id = item['track_id']
        audio = self.audio_embs[str(track_id)]
        return {
            "audio":audio, 
            "track_id":track_id, 
            "tags":tags, 
            "binary":binary, 
            "text":text
            }

    def __getitem__(self, index):
        if (self.split=='TRAIN') or (self.split=='VALID'):
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)
            
    def __len__(self):
        return len(self.fl)
