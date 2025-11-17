import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from PIL import Image


import numpy as np
import json
import os 
import random

class DatasetiCurb(Dataset):    
    r'''
    DataLoader for sampling. Iterate the aerial images dataset
    '''
    def __init__(self,args, mode="valid"):
        #
        assert mode in {"train", "valid", "test"}
        seq_path = args.seq_dir
        mask_path = args.mask_dir

        ####### preprocess update ##############################

        # uncomment these line if you want use the original tiff images
        ########################################################
        # image_path = './dataset/cropped_tiff'
        ########################################################

        orientation_path = args.ori_dir

        if args.test:
            mode = 'test'

        #comment these lines if you want use the original tiff images
        ########################################################
        feature_path_base = "./dataset_manhattan/precomputed_features"

        if mode == 'train':
            image_path = os.path.join(feature_path_base, 'train')
        elif mode == 'valid':
            image_path = os.path.join(feature_path_base, 'valid')
        else: # mode == 'test'
            image_path = os.path.join(feature_path_base, 'test')
        ########################################################
        
        seq_list, mask_list, image_list, ori_list = load_datadir(seq_path,mask_path,image_path,orientation_path,mode)
        self.args = args
        self.seq_len = len(seq_list)
        self.image_list = image_list
        self.seq_list = seq_list
        self.mask_list = mask_list
        self.ori_list = ori_list
        self.mode = mode
    
    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len
    
    def __getitem__(self, idx):
        seq, seq_lens, init_points, end_points = load_seq(self.seq_list[idx],self.args,self.mode)
        image_name = self.seq_list[idx]
        if self.mode=='train':
            tiff, mask, ori = load_image(self.image_list[idx],self.mask_list[idx],self.ori_list[idx])
            return seq, seq_lens, tiff, mask, ori, image_name, init_points, end_points
        else:
            tiff, mask = load_image(self.image_list[idx],self.mask_list[idx])
            return seq, seq_lens, tiff, mask, mask, image_name, init_points, end_points
        

class DatasetDagger(Dataset):
    r'''
    DataLoader for training. Iterate the aggregated DAgger Buffer.
    '''
    def __init__(self,data):
        self.data = data
        self.seq_len = len(data)

    def __len__(self):
        r"""
        :return: data length
        """
        return self.seq_len

    def __getitem__(self, idx):
        cat_tiff = self.data[idx]['cropped_feature_tensor']
        v_now = torch.FloatTensor(self.data[idx]['v_now'])
        v_previous = torch.FloatTensor(self.data[idx]['v_previous'])
        gt_coord = torch.FloatTensor([self.data[idx]['gt_coord']])
        gt_stop_action = torch.LongTensor([self.data[idx]['gt_stop_action']])
        return cat_tiff, v_now, v_previous, gt_coord, gt_stop_action

def load_datadir(seq_path,mask_path,image_path,ori_path,mode):
    with open('./dataset_manhattan/data_split.json','r') as jf:
        json_list = json.load(jf)
    train_list = json_list['train']
    test_list = json_list['test']
    val_list = json_list['valid']

    if mode == 'valid':
        json_list = [x+'.json' for x in val_list][:150]
    elif mode == 'test':
        json_list = [x+'.json' for x in test_list]
        random.shuffle(json_list)
    else:
        json_list = [x+'.json' for x in train_list]

    seq_list = []
    image_list = []
    mask_list = []
    ori_dir = []
    for jsonf in json_list:
        seq_list.append(os.path.join(seq_path,jsonf))
        mask_list.append(os.path.join(mask_path,jsonf[:-4] + 'png'))
        ori_dir.append(os.path.join(ori_path,jsonf[:-4] + 'png'))

        #uncomment these line if you want use the original tiff images
        ########################################################
        # image_list.append(os.path.join(image_path,jsonf[:-4]+'tiff'))
        ########################################################

        #comment these lines if you want use the original tiff images
        ########################################################
        image_list.append(os.path.join(image_path,jsonf[:-4]+'pt'))
        ########################################################

    return seq_list, mask_list, image_list, ori_dir
    
def load_seq(seq_path, args, mode):
    r''' 
    Load the dense sequence of the current image. It may contains the vertices of multiple boundary instances.
    '''
    with open(seq_path) as json_file:
        load_json = json.load(json_file)
        data_json = load_json

    seq_lens = []
    end_points = []
    init_points = []
    for area in data_json:
        seq_lens.append(len(area['seq']))
        if args.gt_init_vertex or not args.test:
            end_points.append(area['end_vertex'])
            init_points.append(area['init_vertex'])

    if not args.gt_init_vertex and args.test:
        
        # --- MODIFICA CRITICA: Estrai solo il nome del file ---
        # 1. Usa os.path.basename() per ottenere SOLO il nome del file (es. SN3_roads...json)
        filename_only = os.path.basename(seq_path) 
        
        # 2. Usa il nome del file per costruire il percorso corretto nella directory di destinazione
        full_init_vertex_path = os.path.join(args.init_vertex_dir, filename_only)
        
        with open(full_init_vertex_path, 'r') as jf:
            init_points = json.load(jf)
    seq = np.zeros((len(seq_lens),max(seq_lens),2))
    for idx,area in enumerate(data_json):
        seq[idx,:seq_lens[idx]] = [x[0:2] for x in area['seq']]
    # seq = torch.FloatTensor(seq)

    return seq, seq_lens, init_points, end_points

def load_image(image_path,mask_path,ori_path=None):

    #uncomment these lines if you want use the original tiff images
    ##########################################################
    #img = Image.open(image_path)
    #img = tvf.to_tensor(img)
    ##########################################################

    #comment these line if you want use the original tiff images
    ##########################################################
    img = torch.load(image_path)
    ##########################################################
    
    assert img.shape[1] == img.shape[2]
    mask_img = Image.open(mask_path).convert('L') 
    mask = np.array(mask_img)
    mask = mask / 255
    if ori_path:
        ori_img = Image.open(ori_path).convert('L')
        ori = np.array(ori_img) 

        return img, mask, ori
    else:
        
        return img, mask

