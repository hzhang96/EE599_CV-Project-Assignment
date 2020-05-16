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

def create_dataset(Config):
    # map id to set_id
    root_dir = Config['root_path']
    train_jason_file = open(osp.join(root_dir, Config['train_jason_file']), 'r')
    train_json = json.load(train_jason_file)
    image_dir = osp.join(root_dir, 'images')
        
    id_to_setid = preprocessing(train_json)
            
    # create X1, X2, y pairs
    files = os.listdir(image_dir)
    X_true = [];  X_false = []; y_true = []; y_false = []
    i = 0
    for x1 in tqdm(files):
        j = 0
        for x2 in files:
            if x1[:-4] in id_to_setid and x2[:-4] in id_to_setid:
                if if_compatible(id_to_setid[x1[:-4]], id_to_setid[x2[:-4]]):
                    X_true.append([x1, x2])
                    y_true.append(1)
                else:
                    X_false.append([x1, x2])
                    y_false.append(0)
                j+=1
                if j == 1000:
                        break
        i+=1
        if i == 1000:
            break
    return X_false, y_false, X_true, y_true

def preprocessing(train_json):
    result = {}
    for i in train_json:
        for j in i['items']:
            if  j['item_id'] in result.keys():
                result[j['item_id']].append(i['set_id']) 
            else:
                result[j['item_id']] = [i['set_id']]
    return result

def if_compatible(a, b):
    for i in a:
        if i in b:
            return True
    return False 
    


if __name__=='__main__':
    X_false, y_false, X_true, y_true  = create_dataset(Config)
    data = {}
    data['X_false'] = X_false
    data['y_false'] = y_false
    data['X_true'] = X_true
    data['y_true'] = y_true
   
    jsObj = json.dumps(data)
 
    fileObject = open('pair_data.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
