import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config
import h5py


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()



    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([300, 300]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    
    def create_dataset(self):
        # import data pair file
        path = Config['data_pair_file']
        p = Config['data_balance']
        X, y = data_balancer(path, p)
        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        output_txt("X_train", 'Q3_input.txt')
        output_txt(X_train, 'Q3_input.txt')
        output_txt("y_train", 'Q3_input.txt')
        output_txt(y_train, 'Q3_input.txt')
        output_txt("X_test", 'Q3_input.txt')
        output_txt(X_test, 'Q3_input.txt')
        output_txt("y_test", 'Q3_input.txt')
        output_txt(y_test, 'Q3_input.txt')
        
        
        return X_train, X_test, y_train, y_test, max(y) + 1



    

#########################################################
#helper function
def output_txt(a, b, new_line = True):
    fileObject = open(b, 'a')
    for i in a:
        fileObject.write(str(i))
        if new_line :
            fileObject.write('\n')
    if new_line == False:
        fileObject.write('\n')
    fileObject.close()  
    
    
def preprocessing(train_json):
    result = {}
    for i in train_json:
        for j in i['items']:
            if  j['item_id'] in result.keys():
                result[j['item_id']].append(i['set_id']) 
            else:
                result[j['item_id']] = [i['set_id']]
    return result

    
def merge(x, y):
    
    width, height = x.size
    result = Image.new(x.mode, (2 * width, height))
    result.paste(y, box=(width, 0))
    return result

def if_compatible(a, b):
    for i in a:
        if i in b:
            return True
    return False 

def data_balancer(path, p):
    jason_file = open(path, 'r')
    data = json.load(jason_file)
    length = len(data['y_true'])
    X = data['X_false'][:length*p] + data['X_true']
    y = data['y_false'][:length*p] + data['y_true']
    return X, y
#########################################################

class polyvore_train(Dataset):
    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path1 = osp.join(self.image_dir, self.X_train[item][0])
        file_path2 = osp.join(self.image_dir, self.X_train[item][1])
        p1 = Image.open(file_path1)
        p2 = Image.open(file_path2)
        p_merge = merge(p1, p2)
        return self.transform(p_merge),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path1 = osp.join(self.image_dir, self.X_test[item][0])
        file_path2 = osp.join(self.image_dir, self.X_test[item][1])
        p1 = Image.open(file_path1)
        p2 = Image.open(file_path2)
        p_merge = merge(p1, p2)
        return self.transform(p_merge),self.y_test[item]


def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size







  
