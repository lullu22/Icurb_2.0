import time
import numpy as np
import torch
import os
import math
import random
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, L1Loss, BCEWithLogitsLoss
from scipy.spatial import cKDTree
import torch.nn.functional as F
from PIL import Image, ImageDraw
import math

from utils.dataset import DatasetiCurb,DatasetDagger
from models.models_encoder import FPN
from models.models_decoder import DecoderCoord, DecoderStop

class FrozenClass(): # to prevent adding new attributes to the class
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True


# create the environment for training and testing iCurb #

class Environment(FrozenClass): 
    def __init__(self,args): # initialize the environment
        self.args = args # store the arguments
        self.crop_size = 63 # crop size for the attention region

        # self.records_dir = 'r{}_f{}'.format(args.r_exp,args.f_exp)
        self.agent = Agent(self) # create the agent
        self.network = Network(self) # create the network

        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len() # total number of training images
        self.graph_record = torch.zeros(1,1,1000,1000).to(args.device) # record the predicted graph for the current image
        self.time_start = time.time()

        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.DAgger_buffer_size = 2048
        self.DAgger_buffer = []
        self.DAgger_buffer_index = 0
        self.setup_seed(20)

        self._freeze() # freeze the class to prevent adding new attributes

    def setup_seed(self,seed): # set the random seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def init_image(self,valid=False): # initialize the environment for a new image
        self.graph_record = torch.zeros(1,1,1000,1000).to(self.args.device)
    
    def update_DAgger_buffer(self,data): # update the DAgger buffer with new data
        if len(self.DAgger_buffer) < self.DAgger_buffer_size: # if the buffer is not full 
            self.DAgger_buffer.append(data) 
            self.DAgger_buffer_index += 1
        else:
            if self.DAgger_buffer_index >= self.DAgger_buffer_size: # if the buffer is full, start overwriting from the beginning
                self.DAgger_buffer_index = 0
            self.DAgger_buffer[self.DAgger_buffer_index] = data
            self.DAgger_buffer_index += 1
            

    def update_graph(self,start_vertex,end_vertex,graph): # update the graph with a new edge
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        instance_vertices = []
        p = start_vertex
        d = end_vertex - start_vertex
        N = np.max(np.abs(d))
        graph[:,:,start_vertex[0],start_vertex[1]] = 1
        graph[:,:,end_vertex[0],end_vertex[1]] = 1
        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                graph[:,:,int(round(p[0])),int(round(p[1]))] = 1



    def expert_restricted_exploration(self,pre_coord,cropped_feature_tensor,thr = 15): # definition of the expert policy for restricted exploration
        r'''
            Based on the cropped region and coord prediction, find the vertex among the 
            gt pixels as the expert demonstration to train iCurb.
        '''
        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy()) # convert the predicted coord to the image coordinate system

        # next vertex for updating
        v_next = pre_coord                # set the next vertex to the predicted coord
        self.agent.taken_stop_action = 0  # initialize the taken stop action
        self.agent.gt_stop_action = 0     # initialize the ground truth stop action

        # generate the expert demonstration for coord prediction

        if self.agent.tree is not None: # if there are gt pixels in the cropped region
            dd, ii = self.agent.tree.query([pre_coord],k=[1]) # find the clostest gt pixel to the predicted coord
            dd = dd[0] # distance between the predicted coord and the closest gt pixel
            ii = ii[0] # index of the closest gt pixel
            gt_coord = self.agent.candidate_label_points[int(ii)].copy() # get the closest gt pixel
            gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)  # find the index of the closest gt pixel in the instance vertices
            self.agent.ii = gt_index # update the index of the current vertex in the instance vertices
            if dd < thr: # if the distance is less than the threshold, use the predicted coord to update the graph
                self.update_graph(self.agent.v_now,pre_coord,self.graph_record)  # update the graph with the predicted coord
            else:
                # expert demonstration for updating
                self.update_graph(self.agent.v_now,gt_coord,self.graph_record) # update the graph with the closest gt pixel
                v_next = gt_coord.copy()

        else: # if there are no gt pixels in the cropped region

            self.agent.gt_stop_action = 1 # set the ground truth stop action to 1
            self.agent.taken_stop_action = 1 # take the stop action

            # whether reach the end vertex
            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<15): # if the agent is close to the end vertex
                gt_coord = self.agent.end_vertex.copy() # set the gt coord to the end vertex
                self.update_graph(self.agent.v_now,gt_coord,self.graph_record) # update the graph with the end vertex
                self.agent.candidate_label_points = [gt_coord] # set the candidate label points to the end vertex

        # save data
        v_now_save = [x/1000 for x in self.agent.v_now] # normalize the current vertex
        v_previous_save = [x/1000 for x in self.agent.v_previous] # normalize the previous vertex
        # store the data in a dictionary, where 'crop_info' contains the information of the cropped region and 'gt_stop_action' is the ground truth stop action, 'v_now' is the current vertex, 'v_previous' is the previous vertex, 'cropped_feature_tensor' is the cropped feature tensor, 'ahead_vertices' are the gt pixels in the cropped region
        stored_data =  {'ahead_vertices':self.agent.candidate_label_points,
            'cropped_feature_tensor':cropped_feature_tensor,
            'crop_info':self.agent.crop_info,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}
        
        self.update_DAgger_buffer(stored_data) # update the DAgger buffer with the new data

        # update
        self.agent.v_previous = self.agent.v_now # update the previous vertex to the current vertex
        self.agent.v_now = v_next # update the current vertex to the next vertex
        


    def expert_free_exploration(self,pre_coord,cropped_feature_tensor): # definition of the expert policy for free exploration
        r'''
            Based on the cropped region and coord prediction, find the vertex among the 
            gt pixels as the expert demonstration to train iCurb.
        '''

        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy()) # convert the predicted coord to the image coordinate system

        # next vertex for updating
        v_next = pre_coord 

        # initialization
        self.agent.taken_stop_action = 0 
        self.agent.gt_stop_action = 0
        
        # generate the expert demonstration for coord prediction
        if self.agent.tree: # if there are gt pixels in the cropped region

            dd, ii = self.agent.tree.query([pre_coord],k=[1]) # find the clostest gt pixel to the predicted coord
            dd = dd[0] # distance between the predicted coord and the closest gt pixel
            ii = ii[0] # index of the closest gt pixel
            gt_coord = self.agent.candidate_label_points[int(ii)].copy() # get the closest gt pixel
            gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1) # find the index of the closest gt pixel in the instance vertices
            
            # update history points (past points)
            if (self.agent.ii <= self.agent.pre_ii): # if the current index is less than or equal to the previous index, it means the agent is going backward
                self.agent.gt_stop_action = 1 # set the ground truth stop action to 1

            self.agent.pre_ii = self.agent.ii # update the previous index to the current index
            self.agent.ii = gt_index # update the current index to the index of the closest gt pixel
            
        else:

            self.agent.gt_stop_action = 1 # set the ground truth stop action to 1

            # whether reach the end vertex
            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<15): # if the agent is close to the end vertex
                self.agent.taken_stop_action = 1 # take the stop action
                gt_coord = self.agent.end_vertex.copy() # set the gt coord to the end vertex
                self.agent.candidate_label_points = [gt_coord] # set the candidate label points to the end vertex
            
        self.update_graph(self.agent.v_now,v_next,self.graph_record) # update the graph with the predicted coord

        # save data
        v_now_save = [x/1000 for x in self.agent.v_now] # normalize the current vertex
        v_previous_save = [x/1000 for x in self.agent.v_previous] # normalize the previous vertex
        # store the data in a dictionary, where 'crop_info' contains the information of the cropped region and 'gt_stop_action' is the ground truth stop action, 'v_now' is the current vertex, 'v_previous' is the previous vertex, 'cropped_feature_tensor' is the cropped feature tensor, 'ahead_vertices' are the gt pixels in the cropped region
        stored_data =  {'ahead_vertices':self.agent.candidate_label_points,
            'cropped_feature_tensor':cropped_feature_tensor,
            'crop_info':self.agent.crop_info,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}

        self.update_DAgger_buffer(stored_data) # update the DAgger buffer with the new data
        
        # update
        self.agent.v_previous = self.agent.v_now # update the previous vertex to the current vertex
        self.agent.v_now = v_next # update the current vertex to the next vertex


# --------------------------------------------------------------------------------------
# Expert Policies for iCurb Training: Restricted vs Free Exploration
#
# The functions `expert_restricted_exploration` and `expert_free_exploration` define
# expert behaviors used to generate supervised training samples for the iCurb model.
#
# - `expert_restricted_exploration` is used during restricted exploration, where the agent
#   follows annotated curb-line sequences. It compares the predicted coordinate to the
#   ground truth (GT) and either accepts it (if close enough) or corrects it using the
#   nearest GT point. It does not check for backward movement.
#
# - `expert_free_exploration` is used during free exploration, where the agent moves
#   autonomously. In addition to checking proximity to GT points, it also monitors
#   whether the agent is moving backward (based on vertex indices). If so, it sets
#   `gt_stop_action = 1` to indicate that the agent should stop.
#
# Both functions convert predicted coordinates to image space, update the graph,
# and store training samples in the DAgger buffer, including stop action labels.
# --------------------------------------------------------------------------------------


class Agent(FrozenClass): # define the agent class for iCurb

    def __init__(self,env): # initialize the agent
        self.env = env 
        self.args = env.args

        # state
        self.v_now = [0,0] # current vertex
        self.v_previous = [0,0] # previous vertex
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0

        # instance information of the current curb instance
        self.instance_vertices = np.array([]) # vertices of the current curb instance
        self.candidate_label_points = np.array([]) # candidate label points in the cropped region
        self.tree = None # KD-tree for fast nearest neighbor search
        self.crop_info = [] # information of the cropped region
        self.init_vertex = [0,0] # initial vertex of the current curb instance
        self.end_vertex = [0,0] # end vertex of the current curb instance
        self.ii = 0 # index of the current vertex in the instance vertices
        self.pre_ii = -1 # index of the previous vertex in the instance vertices
        self._freeze()

    def init_agent(self,init_vertex):   # initialize the agent for a new curb instance
        self.taken_stop_action = 0      # reset the taken stop action
        self.gt_stop_action = 0         # reset the ground truth stop action
        self.agent_step_counter = 0     # reset the step counter
        self.v_now = init_vertex        # set the current vertex to the initial vertex
        self.v_previous = init_vertex   # set the previous vertex to the initial vertex
        self.ii = 0                     # reset the index of the current vertex in the instance vertices
        self.pre_ii = -1                # reset the index of the previous vertex in the instance vertices
        
    def train2world(self,coord_in,crop_info=None): # convert the predicted coord in the cropped region to the image coordinate system (from normalized coord to image coord)
        
        if crop_info is None: # if crop_info is not provided, use the stored crop_info
            crop_info = self.crop_info  

        crop_size = self.env.crop_size  # crop size
        pre_coord = [int(x*(crop_size//2)) for x in coord_in] # example of computation: 0.1*31 = 3.1 -> 3 (assuming crop_size=63)
        pre_coord[0] += crop_info[4]  # where crop_info[4] is the top-left y coordinate of the cropped region in the image
        pre_coord[1] += crop_info[5]  # where crop_info[5] is the top-left x coordinate of the cropped region in the image
        pre_coord = [max(min(pre_coord[0],crop_info[3]-1),crop_info[1]),max(min(pre_coord[1],crop_info[2]-1),crop_info[0])] # ensure the coord is within the cropped region
        return pre_coord

    def crop_attention_region(self,fpn_feature_tensor,val_flag=False): # crop the attention region centering at the current vertex
        r'''
            Crop the current attension region centering at v_now.
        '''
        crop_size = self.env.crop_size # crop size

        # find left, right, up and down positions
        l = self.v_now[1]-crop_size//2 
        r = self.v_now[1]+crop_size//2+1
        d = self.v_now[0]-crop_size//2
        u = self.v_now[0]+crop_size//2+1
        crop_l, crop_r, crop_d, crop_u = 0, self.env.crop_size, 0, self.env.crop_size # initialize the crop positions

        # handle the boundary cases
        if l<0:
            crop_l = -l              # if left position is out of bounds, adjust the crop position
        if d<0:
            crop_d = -d              # if down position is out of bounds, adjust the crop position
        if r>1000:
            crop_r = crop_r-r+1000   # if right position is out of bounds, adjust the crop position
        if u>1000:
            crop_u = crop_u-u+1000   # if up position is out of bounds, adjust the crop position


        crop_l,crop_r,crop_u,crop_d = int(crop_l),int(crop_r),int(crop_u),int(crop_d) # convert to int
        l,r,u,d = max(0,min(1000,int(l))),max(0,min(1000,int(r))),max(0,min(1000,int(u))),max(0,min(1000,int(d))) # ensure the crop positions are within bounds

        self.crop_info = [l,d,r,u,self.v_now[0],self.v_now[1]] # store the crop information

        # cropped feature tensor for iCurb
        cropped_feature_tensor = torch.zeros(1,8,self.env.crop_size,self.env.crop_size) # initialize the cropped feature tensor
        cropped_graph = torch.zeros(1,1,self.env.crop_size,self.env.crop_size) # initialize the cropped graph tensor

        cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r] # crop the feature tensor
        cropped_graph[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record[:,:,d:u,l:r] # crop the graph tensor

        cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach() # concatenate the feature and graph tensors

        # update the gt pixels within the cropped region
        if not val_flag: # if not in validation mode, update the candidate label points and KD-tree (because in validation mode, we do not have gt pixels)ù

            ahead_points = self.instance_vertices[self.ii:] # we take the points starting from the current index ii, and all this points are ahead points (ancora da visitare)
            cropped_ahead_points = (ahead_points[:,0]>=d) * (ahead_points[:,0]<u) * (ahead_points[:,1] >=l) * (ahead_points[:,1] <r) # find the points within the cropped region
            points_index = np.where(cropped_ahead_points==1) # get the indices of the points within the cropped region
            cropped_ahead_points = ahead_points[points_index] # get the points within the cropped region
            self.candidate_label_points = [x for x in cropped_ahead_points if (((x[0] - self.v_now[0])**2 + (x[1]- self.v_now[1])**2)**0.5>15)] # filter the points that are too close to the current vertex (distance > 15 pixels)
            if len(self.candidate_label_points): # if there are candidate label points, build the KD-tree for fast nearest neighbor search
                self.tree = cKDTree(self.candidate_label_points)
            else:
                self.tree = None # if no candidate label points, set the tree to None
        return cropped_feature_tensor.to(self.args.device) # return the cropped feature tensor
    
# --------------------------------------------------------------------------------------
# Agent Class for iCurb
#
# This class defines the behavior and internal state of the iCurb agent, which navigates
# along curb-line instances during training and inference. The agent interacts with the
# environment to:
# - Track its current and previous positions (`v_now`, `v_previous`)
# - Manage stop actions (`taken_stop_action`, `gt_stop_action`)
# - Store curb-line vertices and candidate label points for supervision
# - Crop attention regions from the feature map centered on its current position
# - Convert predicted coordinates from normalized space to image space (`train2world`)
# - Build a KD-tree for fast nearest neighbor search among ground truth points
#
# The agent is reinitialized for each curb instance and maintains step counters and
# crop metadata to support training with DAgger and expert policies.
# --------------------------------------------------------------------------------------
   

class Network(FrozenClass): # define the network class for iCurb
    def __init__(self,env): # initialize the network
        self.env = env
        self.args = env.args

    
        # initialization
        self.encoder = FPN() # FPN encoder
        self.decoder_coord = DecoderCoord(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4) # coordinate decoder
        self.decoder_stop = DecoderStop(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4) # stop action decoder
        self.encoder.to(device=self.args.device)       # move the encoder to the specified device
        self.decoder_coord.to(device=self.args.device) # move the coordinate decoder to the specified device
        self.decoder_stop.to(device=self.args.device)  # move the stop action decoder to the specified device
        print(self.args.device) 

        # tensorboard 
        if not self.args.test: # if not in test mode, create a tensorboard writer
            self.writer = SummaryWriter('./records/tensorboard') # create a tensorboard writer

        
        # ====================optimizer======================= # define the optimizers for the encoder and decoders
        self.optimizer_enc = optim.Adam(list(self.encoder.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)             # encoder optimizer 
        self.optimizer_coord_dec = optim.Adam(list(self.decoder_coord.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay) # coordinate decoder optimizer
        self.optimizer_flag_dec = optim.Adam(list(self.decoder_stop.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)   # stop action decoder optimizer


        # =====================init losses======================= # define the loss functions
        criterion_l1 = L1Loss(reduction='mean') # L1 loss for coordinate regression
        criterion_bce = BCEWithLogitsLoss() # BCE loss for stop action classification
        criterion_ce = CrossEntropyLoss() # Cross-entropy loss for classification tasks (not used in this code)
        self.criterions = {"ce":criterion_ce,'l1':criterion_l1,"bce": criterion_bce} # store the loss functions in a dictionary


        # =====================Load data======================== # load the training and validation datasets
        dataset_train = DatasetiCurb(self.args,mode="train") # training dataset
        dataset_valid = DatasetiCurb(self.args,mode="valid") # validation dataset
        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True,collate_fn=self.iCurb_collate)  # training dataloader 
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.iCurb_collate) # validation dataloader
        print("Dataset splits -> Train: {} | Valid: {}\n".format(len(dataset_train), len(dataset_valid))) # print the dataset sizes
        self.loss = 0 # initialize the loss
        self.best_f1 = 0 # initialize the best F1 score
        #
        self.load_checkpoints() # load the pretrained checkpoints
        self._freeze() 

    def load_checkpoints(self): # load the pretrained checkpoints
        self.encoder.load_state_dict(torch.load('./checkpoints/seg_pretrain.pth',map_location='cpu')) # load the pretrained FPN encoder checkpoint

        if self.args.test: # if in test mode, load the best decoder checkpoints

            self.decoder_coord.load_state_dict(torch.load("./checkpoints/decoder_nodis_coord_best.pth",map_location='cpu')) # load the best coordinate decoder checkpoint
            self.decoder_stop.load_state_dict(torch.load("./checkpoints/decoder_nodis_flag_best.pth",map_location='cpu'))   # load the best stop action decoder checkpoint

            print('=============')
            print('Successfully loading iCurb checkpoints!')
        
        print('=============')
        print('Pretrained FPN encoder checkpoint loaded!')
    
    def train_mode(self):    # set the network to training mode
        self.decoder_coord.train() # set the coordinate decoder to training mode
        self.decoder_stop.train()  # set the stop action decoder to training mode
    
    def val_mode(self):      # set the network to evaluation mode
        self.decoder_coord.eval()  # set the coordinate decoder to evaluation mode
        self.decoder_stop.eval()   # set the stop action decoder to evaluation mode
    
    def train_len(self):     # get the length of the training dataloader
        return len(self.dataloader_train) 

    def val_len(self):       # get the length of the validation dataloader
        return len(self.dataloader_valid)

    def bp(self):            # backpropagation and optimizer step  
        self.optimizer_coord_dec.zero_grad() # zero the gradients for the coordinate decoder optimizer
        self.optimizer_flag_dec.zero_grad()  # zero the gradients for the stop action decoder optimizer
        self.loss.backward()                 # backpropagate the loss
        self.optimizer_flag_dec.step()       # update the stop action decoder parameters
        self.optimizer_coord_dec.step()      # update the coordinate decoder parameters
        self.loss = 0                        # reset the loss

    def save_checkpoints(self,i): # save the best model checkpoints
        print('Saving checkpoints {}.....'.format(i)) 
        torch.save(self.decoder_coord.state_dict(), "./checkpoints/decoder_nodis_coord_best.pth")  # save the best coordinate decoder checkpoint
        torch.save(self.decoder_stop.state_dict(), "./checkpoints/decoder_nodis_flag_best.pth")    # save the best stop action decoder checkpoint


    def DAgger_collate(self,batch): # collate function for DAgger buffer

        r'''
            Collate function for DAgger buffer.
            batch: list of dictionaries {'ahead_vertices','cropped_feature_tensor','crop_info','gt_stop_action','v_now','v_previous'}
        '''

        # variables as list
        crop_info = [x[3].tolist() for x in batch]      # crop information
        cropped_point = [x[4].tolist() for x in batch]  # candidate label points in the cropped region

        # variables as tensor
        cat_tiff = torch.stack([x[0] for x in batch])   # cropped feature tensor
        v_now = torch.stack([x[1] for x in batch])      # current vertex
        v_previous = torch.stack([x[2] for x in batch]) # previous vertex
        gt_stop_action = torch.stack([x[-1] for x in batch]).reshape(-1) # ground truth stop action

        return cat_tiff, v_now, v_previous, crop_info, cropped_point, gt_stop_action

    def iCurb_collate(self,batch): # collate function for iCurb dataset

        r'''
            Collate function for iCurb dataset.
            batch: list of tuples (seq, seq_len, tiff, mask, image_name, init_point, end_point)
        '''

        # variables as numpy
        seq = np.array([x[0] for x in batch])            # sequence of vertices
        mask = np.array([x[3] for x in batch])           # mask for the sequence

        # variables as list
        seq_lens = [x[1] for x in batch]                 # lengths of the sequences
        image_name = [x[4] for x in batch]               # image names
        init_points = [x[5] for x in batch]              # initial points of the sequences
        end_points = [x[6] for x in batch]               # end points of the sequences
        
        # variables as tensor
        tiff = torch.stack([x[2] for x in batch])        # input images

        return seq, seq_lens, tiff, mask, image_name, init_points, end_points

# --------------------------------------------------------------------------------------
# Network Class for iCurb
#
# This class defines the core neural architecture and training logic for the iCurb model.
# It includes:
# - An FPN encoder for extracting multi-scale features from input images.
# - Two decoders:
#     - `decoder_coord` for predicting the next coordinate to visit.
#     - `decoder_stop` for predicting whether the agent should stop.
# - Optimizers for each module and loss functions for training:
#     - L1 loss for coordinate regression.
#     - Cross-entropy loss for stop action classification.
# - Dataloaders for training and validation using the DatasetiCurb class.
# - TensorBoard integration for logging training metrics.
# - Checkpoint loading and saving for model persistence.
#
# The class also provides utility methods to switch between training and evaluation modes,
# perform backpropagation, and collate data batches for both iCurb and DAgger training.
# --------------------------------------------------------------------------------------
