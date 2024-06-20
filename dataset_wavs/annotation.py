import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset
from ttmrpp_nc.utils.audio_utils import int16_to_float32, float32_to_int16, load_audio, STR_CH_FIRST


class Annotation_Dataset(Dataset):
    def __init__(self, data_path, split, sr, duration, num_chunks):
        self.data_path = data_path
        self.split = split
        self.sr = sr
        self.input_length = int(sr * duration)
        self.num_chunks = num_chunks
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
            annotation = {str(item['track_id']).lower(): item for item in annotation_data}
        
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
        

    def audio_load(self, path: str) -> np.ndarray:
        audio_path = os.path.join(self.data_path, "Annotation", "Audio", path)
        audio_filename = audio_path + ".npy"
        
        try:
            audio, _ = load_audio(
                path=audio_filename,
                ch_format=STR_CH_FIRST,
                sample_rate=self.sr,
                downmix_to_mono=False
            )

            if audio is None or audio.shape[0] < self.input_length:
                return None

            if len(audio.shape) == 2:
                audio = np.mean(audio, axis=0)  # Downmix to mono if still in 2D form
            if audio.ndim != 1:
                raise ValueError("Audio data is not 1-dimensional after downmixing.")

            audio = int16_to_float32(float32_to_int16(audio.astype('float32')))  # for float32 loader

            hop = (audio.shape[0] - self.input_length) // self.num_chunks
            audio_chunks = np.stack([audio[i * hop: i * hop + self.input_length] for i in range(self.num_chunks)]).astype('float32')

            return audio_chunks
        
        except Exception as e:
            print(f"Error loading audio for {path}: {e}")
            return None

    
    def tag_to_binary(self, text):
        binary = np.zeros([len(self.list_of_label),], dtype=np.float32)
        if isinstance(text, str):
            binary[self.tag_to_idx[text]] = 1.0
        elif isinstance(text, list):
            for tag in text:
                binary[self.tag_to_idx[tag]] = 1.0
        return binary
    
    def __getitem__(self, index):
        if index >= len(self.fl):
            raise IndexError("Index out of range.")
        
        item = self.fl[index]
        tag_list = item['tag']
        binary = self.tag_to_binary(tag_list)
        text = ", ".join(tag_list)
        tags = self.list_of_label
        track_id = item['track_id']
        
        audio = self.audio_load(track_id)
        if audio is None:
            new_index = (index + 1) % len(self.fl)
            if new_index == index:
                raise ValueError("No valid audio found in the dataset.")
            return self.__getitem__(new_index)
        
        return {
            "audio": audio,
            "track_id": track_id,
            "tags": tags,
            "binary": binary,
            "text": text
        }

    def __len__(self):
        return len(self.fl)

