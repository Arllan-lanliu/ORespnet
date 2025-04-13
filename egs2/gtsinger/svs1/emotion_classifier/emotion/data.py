from collections import Counter
import torch
import os
import json

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.uids = list(data.keys())
        self.wavs = data
        self.labels = labels
        self.label_to_idx = {emotion: i for i, emotion in enumerate(set(labels.values()))} 

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        wavpath = self.wavs[uid]
        emotion = self.labels[uid]
        label_idx = self.label_to_idx[emotion] 
        return wavpath, label_idx
    
    def get_label(self):
        return self.labels

def count_label(msg, dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.labels[dataset.dataset.uids[idx]] for idx in dataset.indices]
    else:
        labels = list(dataset.get_label().values())
    
    label_counts = Counter(labels)
    print(f"[Process Data] {msg} data: happy ({label_counts['happy']}), sad({label_counts['sad']})")

def get_emotion_label(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data[0]['emotion'] == 'hapsadpy' or data[0]['emotion'] == "hsadappy":
        return 'sad'
    else:
        return data[0]['emotion'] 

def process_data(basedir):
    total_data = {}
    total_label = {}
    for root, dirs, files in os.walk(basedir):
        if not dirs:
            for file in files:
                filepath = os.path.join(root, file)
                plist = filepath.strip('/').split('/')
                index = plist.index('GTSinger')
                if len(plist) != 10 or plist[index+5] == "Paired_Speech_Group" or plist[index+6][5] != "w":
                    continue

                emotion_label = get_emotion_label(os.path.splitext(filepath)[0] + ".json")
                utt_id = '_'.join(plist[index:])[:-4] 
                total_data[utt_id] = filepath
                total_label[utt_id] = emotion_label
    print(set(total_label.values()))

    return EmotionDataset(total_data, total_label)
    