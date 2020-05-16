import numpy as np
import os
import os.path as osp
import argparse

Config ={}
Config['root_path'] = r'D:\StudyMaterial\USC\Semester3_20spring\EE_599_Deep learning\polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['train_jason_file'] = 'train.json'
Config['checkpoint_path'] = ''
Config['data_pair_file'] = 'pair_data.json'
Config['data_balance'] = 5

Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 5
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5