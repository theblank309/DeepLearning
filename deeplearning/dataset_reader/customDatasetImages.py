import os
from PIL import Image

import torch
from torch.utils.data import Dataset


class CustomDatasetImages(Dataset):

    def __init__(self, root_dir, transform) -> None:

        self.all_files, self.all_labels, self.labels_name = self.__get_data(root_dir)
        self.transform = transform

    def __len__(self):
        
        return len(self.all_files)
    
    def __getitem__(self, index):
        
        img = Image.open(self.all_files[index])
        label = torch.tensor(self.all_labels[index])
        
        if self.transform:
            img = self.transform(img)

        if type(img) != "torch.tensor":
            img = torch.tensor(img)

        return (img,label)

    def __get_data(self, root_dir):

        files = []
        labels = []
        label_names = {}

        itr = 0
        for dir in os.listdir(root_dir):
            dir_path = os.path.join(root_dir,dir)
            files.extend([os.path.join(dir_path, file) for file in os.listdir(dir_path)])

            if dir not in label_names.keys():
                label_names[dir] = itr
                itr += 1
            labels.extend([label_names[dir]]*len(os.listdir(dir_path)))

        return files, labels, label_names

