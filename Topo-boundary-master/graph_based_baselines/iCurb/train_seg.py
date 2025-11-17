import argparse
import json
import os
import json
import shutil
import pickle
#
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
#
from models.models_encoder import *
from arguments import *


from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F

### change when we want try new experiment #######
#new_name = "PMM-NY_efficientnet_b4_gaussian1.8"
##################################################

############################## Gaussian Loss Implementation ###############################

class GaussianDistanceLoss(nn.Module):
    def __init__(self, sigma=2.0, reduction="mean"):
        super(GaussianDistanceLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction 

    def forward(self, inputs, targets, already_gaussian=False):
        """
        inputs  : predizione raw logit [B,1,H,W]
        targets : ground truth [B,1,H,W]
        already_gaussian : True se il GT è già in forma gaussiana (es. endpoint)
        """
        B, _, H, W = targets.shape
        device = targets.device

        if already_gaussian:
            # Se il GT è già gaussiano, lo uso direttamente
            heatmaps = targets
        else:
            # Se GT è binario (es. maschere strada), calcolo distance transform
            targets_np = targets.detach().cpu().numpy() 
            heatmaps = []
            for b in range(B):
                gt = targets_np[b,0].astype(np.uint8)
                if gt.sum() > 0:
                    dist_map = distance_transform_edt(1-gt)
                    heatmap = np.exp(-(dist_map**2)/(2*self.sigma**2))
                else:
                    heatmap = np.zeros((H,W), dtype=np.float32)
                heatmaps.append(torch.from_numpy(heatmap).to(device))
            heatmaps = torch.stack(heatmaps).unsqueeze(1).float()

        # Pred sigmoid
        pred = torch.sigmoid(inputs).float()

        # Loss MSE mean squared error
        loss = F.mse_loss(pred, heatmaps, reduction=self.reduction)
        return loss

###########################################################################################



class dataset(Dataset):
    def __init__(self,args,valid=False):
        # train the network with pretrain patches

        ## normal dataset ## 
        #with open('./dataset/data_split.json','r') as jf:
        #    json_list = json.load(jf)['pretrain']
        #####################

        ## PPM-NY dataset ##
        #with open('./dataset_PMM-NY/data_split.json','r') as jf:
        #    json_list = json.load(jf)['pretrain']
        #####################
        
        ## manhattan dataset ##
        with open('./dataset_manhattan/data_split.json','r') as jf:
            json_list = json.load(jf)['pretrain']
        #####################

        self.file_list = json_list
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the training data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    #we use the following code for resnet, and efficientnet_b4 models ##
    
   
    def __getitem__(self,idx):
            tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
            mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
            return tiff,mask
    
    """
    ####################################################################

    # we use the following code for swin transformer large model ##
    def __getitem__(self,idx):
        
        tiff_img = Image.open(self.tiff_list[idx])
        mask_img = Image.open(self.mask_list[idx])

        target_size = (1008,1008)

        tiff = tvf.resize(tiff_img, target_size, interpolation = tvf.InterpolationMode.BICUBIC) # we use bicubic for tiff images for maintaining better quality
        mask = tvf.resize(mask_img, target_size, interpolation = tvf.InterpolationMode.NEAREST) # we use nearest for mask images to avoid interpolation artifacts
        tiff = tvf.to_tensor(tiff)
        mask = tvf.to_tensor(mask)
        return tiff,mask
    #################################################################
    """

class valid_dataset(Dataset):
    def __init__(self,args):

        ## normal dataset ##
        #with open('./dataset/data_split.json','r') as jf:
        #    json_list = json.load(jf) 
        #####################

        ## PPM-NY dataset ##
        #with open('./dataset_PMM-NY/data_split.json','r') as jf:
        #    json_list = json.load(jf)
        #####################

        ## manhattan dataset ##
        with open('./dataset_manhattan/data_split.json','r') as jf:
            json_list = json.load(jf)
        #####################

        self.file_list = json_list['valid'][:500]
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the valid data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)
    
    # we use this code for resnet and efficentnet_b4 models 
    def __getitem__(self,idx):
        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        name = self.file_list[idx]
        return tiff,mask,name 
    

    """
    
    # we use this code for swin trasformer large model 
    def __getitem__(self,idx):
        
        tiff_img = Image.open(self.tiff_list[idx])
        mask_img = Image.open(self.mask_list[idx])
        name = self.file_list[idx]

        target_size = (1008,1008)

        tiff = tvf.resize(tiff_img, target_size, interpolation = tvf.InterpolationMode.BICUBIC) # we use bicubic for tiff images for maintaining better quality
        mask = tvf.resize(mask_img, target_size, interpolation = tvf.InterpolationMode.NEAREST) # we use nearest for mask images to avoid interpolation artifacts
        tiff = tvf.to_tensor(tiff)
        mask = tvf.to_tensor(mask)
        return tiff,mask,name

    """

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

################################## Training ###############################################

def train(args,epoch,net,dataloader,train_len,optimizor,criterion,writer,valid_dataloader,valid_len):
    net.train()
    counter = 0
    best_f1 = 0
    for idx,data in enumerate(dataloader):
        img, mask= data
        img, mask= img.to(args.device), mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        predictions,_ = net(img)



        # Semplifica il calcolo della loss
        if args.loss_type == 'bce':
            loss = criterion(predictions, mask)
        elif args.loss_type == 'gaussian':
            loss = criterion(predictions, mask, already_gaussian=False)


        optimizor.zero_grad()
        loss.backward()
        optimizor.step()

        print('Epoch: {}/{} || batch: {}/{} || loss:{}'.format(epoch,args.epochs,idx,train_len,round(loss.item(),3)))
        writer.add_scalar('train/loss',loss.item(),counter + train_len*epoch)
        counter += 1
        if idx % (train_len-1) == 0 and idx:
            f1 = val(args,epoch,net,valid_dataloader,counter + train_len*epoch,valid_len,writer)
            if f1 > best_f1:
                #whitout new_name
                #torch.save(net.state_dict(), "./checkpoints/seg_pretrain.pth")
                #with experiment name
                exp_name = args.experiment_name
                torch.save(net.state_dict(), "./checkpoints/seg_pretrain_{}.pth".format(exp_name))
                
                best_f1 = f1

############################## Validation ###################################################

def val(args,epoch,net,dataloader,ii,val_len,writer,mode=0):


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

        if gt_points.ndim !=2 or gt_points.shape[0]==0:
            return None, None,None
        
        if graph_points.ndim !=2 or graph_points.shape[0]==0:
            return 0, 0,0
        

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
    f1_sum = 0.0 
    valid_samples = 0
    f1_ave = 0
    for idx,data in enumerate(dataloader):
        img, mask, name = data
        img, mask = img.to(args.device), mask[0,0,:,:].cpu().detach().numpy()


        with torch.no_grad():
            pre_segs,_ = net(img)
            pre_segs = torch.sigmoid(pre_segs[0,0,:,:]).cpu().detach().numpy()

            #without new_name
            #Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save('./records/seg/valid/{}.png'.format(name[0]))

            #with new_name and new folder for different experiments
            exp_name = args.experiment_name
            subfolder_name = f"seg_valid_{exp_name}"
            full_path = os.path.join('./records/seg/valid', subfolder_name)
            os.makedirs(full_path, exist_ok=True)
            Image.fromarray(pre_segs/np.max(pre_segs)*255).convert('RGB').save(os.path.join(full_path, '{}.png'.format(name[0])))
            
           
            pre_segs = (pre_segs>0.2)
            prec, recall, f1 = eval_metric(pre_segs,mask)
            if f1 is None : 
                print('Validation:{}/{} || Image:{}/{} || No GT pixels, skipped.'.format(epoch,args.epochs,idx,val_len))
                continue

            f1_sum += f1
            valid_samples += 1

            print('Validation:{}/{} || Image:{}/{} || Precision/Recall/f1:{}/{}/{}'.format(epoch,args.epochs,idx,val_len,round(prec,3),round(recall,3),round(f1,3)))

    if valid_samples >0:
        f1_ave = f1_sum/valid_samples

            

    print('Validation Summary:{}/{} || Average F1:{}'.format(epoch,args.epochs,round(f1_ave,3)))
    writer.add_scalar('val_F1',f1_ave,ii)
    return f1_ave

############################## Main Function ###############################################

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    #loss selection
    if args.loss_type == 'bce':
        criterion_instance = nn.BCEWithLogitsLoss()
        run_name = 'bce_loss'
    elif args.loss_type == 'gaussian':
        criterion_instance = GaussianDistanceLoss(sigma=args.gaussian_sigma)
        run_name = 'gaussian_loss_sigma_{}'.format(args.gaussian_sigma)
        

    update_dir_seg(args)
    parser = argparse.ArgumentParser()
    device = args.device
    # load data
    train_dataset = dataset(args)
    valid_dataset = valid_dataset(args)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset,batch_size=1,shuffle=False)
    train_len = len(train_dataloader)
    valid_len = len(valid_dataloader)
    # network
    net = FPN()
    net.to(device=device)
    optimizor = torch.optim.Adam(list(net.parameters()),lr=1e-3)


    
    #without new_name
    #log_directory = os.path.join('./records/seg/tensorboard', run_name)
    #with new_name and new folder for different experiments
    exp_name = args.experiment_name
    subfolder_name = f"_{exp_name}"
    full_path = os.path.join('./records/seg/tensorboard', subfolder_name)
    os.makedirs(full_path, exist_ok=True)
    
    log_directory = os.path.join(full_path, run_name + '_' + exp_name)
    
    print(f"Salvando i log di TensorBoard in: {log_directory}")
    writer = SummaryWriter(log_directory)
   
    
    for i in range(args.epochs):
        train(args,i,net,train_dataloader,train_len,optimizor,criterion_instance,writer,valid_dataloader,valid_len)