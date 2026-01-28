import argparse
import json
import os
import json
import shutil
import pickle
from scipy.spatial import cKDTree
import scipy
from skimage import measure
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvf
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from models.models_encoder import *
from arguments import *
import glob 


METRICS_ACTIVATION = False


class valid_dataset(Dataset): # for validation and test
    ''' 
    load the validation/test data
    ''' 
    def __init__(self,args): 

        ####### normal_dataset#########################
        json_path = os.path.join(args.dataset_dir, 'data_split.json')
        ###############################################

        print(f"Loading configuration from: {json_path}")

        with open(json_path,'r') as jf:
            json_list = json.load(jf)
        self.file_list = json_list['test']
        
        
        if METRICS_ACTIVATION:
            relative_image_path = os.path.relpath(args.image_dir, './dataset')
            relative_mask_path = os.path.relpath(args.mask_dir, './dataset')
            relative_endpoint_path = os.path.relpath(args.endpoint_dir, './dataset')
        
            print(f"Percorso relativo maschere: '{relative_mask_path}'")
            print(f"Percorso relativo endpoint: '{relative_endpoint_path}'")

            self.tiff_list = [os.path.join(args.dataset_dir, relative_image_path, f'{x}.tiff') for x in self.file_list]
            self.mask_list = [os.path.join(args.dataset_dir, relative_mask_path, f'{x}.png') for x in self.file_list]
            self.endpoint_list = [os.path.join(args.dataset_dir, relative_endpoint_path, f'{x}.png') for x in self.file_list]
        else: 

            relative_image_path = os.path.relpath(args.image_dir, './dataset')
            self.tiff_list = [os.path.join(args.dataset_dir, relative_image_path, f'{x}.tiff') for x in self.file_list]

        
        print('Finish loading the dataset lists {}!'.format(len(self.file_list)))


    def __len__(self):
        return len(self.file_list) 

    def __getitem__(self,idx):

        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx])) 

        if METRICS_ACTIVATION:
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))

        else: 
            mask = torch.zeros((1, tiff.shape[1], tiff.shape[2]))  # dummy mask for compatibility

        name = self.file_list[idx]

        return tiff,mask,name 
     

       
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch)) # remove the None data
    return torch.utils.data.dataloader.default_collate(batch) 


def val(args,net,dataloader,val_len):

    if METRICS_ACTIVATION: 
    
        def eval_metric(seg_result,mask): 
            '''
            Evaluate the predicted image by F1 score during evaluation
            '''
            def tuple2list(t):
                return [[t[0][x],t[1][x]] for x in range(len(t[0]))] 

            skel = skeletonize(seg_result, method='lee')
            gt_points = tuple2list(np.where(mask!=0))
            graph_points = tuple2list(np.where(skel!=0))
            gt_points = np.array(gt_points)
            graph_points = np.array(graph_points)

            if gt_points.ndim != 2 or gt_points.shape[0] == 0:
                return None, None, None
            
            if graph_points.ndim != 2 or graph_points.shape[0] == 0:
                return 0, 0, 0

            graph_acc = 0
            graph_recall = 0
            gt_tree = scipy.spatial.cKDTree(gt_points)
            for c_i,thre in enumerate([5]):
                if len(graph_points):
                    graph_tree = scipy.spatial.cKDTree(graph_points)
                    graph_dds,_ = graph_tree.query(gt_points, k=1)
                    gt_acc_dds,gt_acc_iis = gt_tree.query(graph_points, k=1)
                    graph_recall = len([x for x in graph_dds if x<thre])/len(graph_dds)
                    graph_acc = len([x for x in gt_acc_dds if x<thre])/len(gt_acc_dds)
            r_f = 0
            if graph_acc*graph_recall:
                r_f = 2*graph_recall * graph_acc / (graph_acc+graph_recall)
            return graph_acc, graph_recall,r_f

    net.eval()
    f1_ave = 0
    valid_count = 0
    count_low_f1 = 0
    count_high_f1 = 0

    # initialize metrics #############################
    f1 = 0
    prec = 0
    recall = 0
    ##################################################

    for idx,data in enumerate(dataloader):

        if METRICS_ACTIVATION:
            img, mask, name = data
            img, mask = img.to(args.device), mask[0,0,:,:].cpu().detach().numpy()

        else:
            img, _,  name = data
            img = img.to(args.device)
        
        with torch.no_grad():

            pre_segs_raw, pre_endpoint_raw, _ = net(img)
            pre_segs = torch.sigmoid(pre_segs_raw[0,0,:,:]).cpu().detach().numpy()
            pre_endpoint = torch.sigmoid(pre_endpoint_raw[0,0,:,:]).cpu().detach().numpy()
            
            if METRICS_ACTIVATION:
                pre_segs_thresh = (pre_segs>0.2) 
                prec, recall, f1 = eval_metric(pre_segs_thresh,mask)
        
            if f1 is not None:
                f1_ave = (f1_ave * valid_count + f1) / (valid_count+1)
                valid_count += 1
               
                print('Inference:{}/{} || Precision/Recall/f1:{}/{}/{}'.format(idx,val_len,round(prec,3),round(recall,3),round(f1,3)))
                
                if f1 < 0.5:
                    count_low_f1 += 1
                    #seg_save_path = './records/seg/test/low_f1'
                    #endpoint_save_path = './records/endpoint/test/low_f1'

                    #seg_save_path = './records/seg/test_PMM-NY'
                    seg_save_path = './records/seg/RL_Prova'
                    #seg_save_path_skeleton = './records/seg/test/skeleton_test'
                    #endpoint_save_path = './records/endpoint/test_PMM-NY'
                    endpoint_save_path = './records/endpoint/RL_Prova'

                else:
                    count_high_f1 += 1
                    #seg_save_path = './records/seg/test/high_f1'
                    #endpoint_save_path = './records/endpoint/test/high_f1'
                
                    #seg_save_path = './records/seg/test_PMM-NY'
                    seg_save_path = './records/seg/RL_Prova'
                    #seg_save_path_skeleton = './records/seg/test/skeleton_test'
                    #endpoint_save_path = './records/endpoint/test_PMM-NY'
                    endpoint_save_path = './records/endpoint/RL_Prova'
              
                os.makedirs(seg_save_path, exist_ok=True)
                #os.makedirs(seg_save_path_skeleton, exist_ok=True)
                os.makedirs(endpoint_save_path, exist_ok=True)
               
                Image.fromarray((pre_segs * 255).astype(np.uint8)).convert('RGB').save(os.path.join(seg_save_path, '{}.png'.format(name[0])))
                #Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB').save(os.path.join(seg_save_path, '{}_gt.png'.format(name[0])))
                #Image.fromarray((pre_segs_thresh*255).astype(np.uint8)).convert('RGB').save(os.path.join(seg_save_path_skeleton, '{}_skeleton.png'.format(name[0])))
                Image.fromarray((pre_endpoint * 255).astype(np.uint8)).convert('RGB').save(os.path.join(endpoint_save_path, '{}.png'.format(name[0])))
            
            else:

                print('Inference:{}/{} || Precision/Recall/f1: None/None/None'.format(idx,val_len))
                

    if METRICS_ACTIVATION:
        print('Average F1:{}'.format(round(f1_ave,3)))
        print('low f1 count:{}'.format(count_low_f1))
        print('high f1 count:{}'.format(count_high_f1))

        return f1_ave, count_low_f1, count_high_f1
    
    else:
        print("Validation completed.")

        return
   


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    update_dir_seg(args)

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./dataset', 
                        help='path to the dataset (es. ./dataset_NY)')
    
    # we do again the argument parsing to override the default args
    new_args = parser.parse_args()
    for key, value in vars(new_args).items():
        setattr(args, key, value)
    
    device = args.device

    valid_dataset = valid_dataset(args)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    valid_len = len(valid_dataloader)

    #net = FPN()
    net = FPN(backbone_name='efficientnet_b4')
    net.to(device=device)

   
    # checkpoint loading #

    # 1. specify the checkpoint path
    checkpoint_path = "./checkpoints/seg_pretrain_gaussian_PMM-NY_efficentnet_b4_1.6_RL.pth"
    print(f"Attempting to load checkpoint from: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        # 2. Load the entire file into a variable
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 3. Check if it's the dictionary format (saved from train.py)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 4. Extract ONLY the weights and load them into the model
            net.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Checkpoint (dictionary format) loaded successfully.")
        else:
            # Otherwise, try to load it as if it contained only the weights
            net.load_state_dict(checkpoint)
            print("✅ Checkpoint (legacy format) loaded successfully.")
    else:
        print(f"⚠️ WARNING: Checkpoint not found. Model will start with random weights.")

    # ############################################################### #


    val(args, net, valid_dataloader, valid_len)