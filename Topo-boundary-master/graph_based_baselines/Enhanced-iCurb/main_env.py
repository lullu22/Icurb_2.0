import time
import numpy as np
import torch
from skimage.draw import line as draw_line
import math
import os
import random
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, L1Loss, BCEWithLogitsLoss
from scipy.spatial import cKDTree
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure


from utils.dataset import DatasetiCurb,DatasetDagger
from models.models_encoder import FPN
from models.models_decoder import DecoderCoord, DecoderStop, CombinedDecoder

class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

class Environment(FrozenClass):
    def __init__(self,args):
        self.args = args
        self.crop_size = 63
        self.agent = Agent(self)
        self.network = Network(self)
        # recordings
        self.training_image_number = self.args.epochs * self.network.train_len() # total training image number computed by epochs * training images per epoch
        
        #graph record on GPU
        #######################
        #self.graph_record = torch.zeros(1,1,1000,1000).to(args.device) # graph record for the current image
        #######################

        #graph record on CPU
        ########################
        self.graph_record_np = np.zeros((1, 1, 1000, 1000), dtype=np.float32) 
        ########################


        self.time_start = time.time()
        # ===================parameters===================
        self.training_step = 0
        self.epoch_counter = 0
        self.DAgger_buffer_size = 50000 # DAgger buffer size (default 2048)
        self.DAgger_buffer = []
        self.DAgger_buffer_index = 0
        self.init_point_set = []
        self.setup_seed(20)

        ###############################################
        ###### modifications for dagger training ######
        self.last_loss_coord = 0.0
        self.last_loss_stop = 0.0
        ###############################################
        ###############################################

        self._freeze()

    # set random seed for reproducibility
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # initialize the graph record for each image

    #initialize on GPU
    ##################################
    #def init_image(self,valid=False):
    #    self.graph_record = torch.zeros(1,1,1000,1000).to(self.args.device)
    ##################################

    #initialize on CPU
    ##################################
    def init_image(self,valid=False):
        self.graph_record_np.fill(0) # reset the graph record to zero
    ##################################
    
    ########################## DAgger buffer update #########################################

    def update_DAgger_buffer(self,data):
        if len(self.DAgger_buffer) < self.DAgger_buffer_size: # buffer not full
            self.DAgger_buffer.append(data)
            self.DAgger_buffer_index += 1
        else:
            if self.DAgger_buffer_index >= self.DAgger_buffer_size: # buffer full and index out of range, so reset from 0 
                self.DAgger_buffer_index = 0 # reset index
            self.DAgger_buffer[self.DAgger_buffer_index] = data # replace data in the new index with the new data
            self.DAgger_buffer_index += 1 # update index

    #######################################################################################
            
    ############################# Graph update ############################################

    ## first method for GPU graph update (slow) ##
    ###################################################################################
    #def update_graph(self,start_vertex,end_vertex,graph):
    #    start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])])
    #    end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
    #    instance_vertices = []
    #    p = start_vertex # current position
    #    d = end_vertex - start_vertex # direction vector
    #    N = np.max(np.abs(d)) # number of steps

    #    graph[:,:,start_vertex[0],start_vertex[1]] = 1 # set the start vertex, we set to 1 to indicate the starting point
    #    graph[:,:,end_vertex[0],end_vertex[1]] = 1 # set the end vertex, we set to 1 to indicate the end point

    #    if N: # if the number of steps is not zero ( this means start_vertex != end_vertex)

    #        s = d / (N) # step size

    #       for i in range(0,N): 
    #            p = p + s
    #           graph[:,:,int(round(p[0])),int(round(p[1]))] = 1 # set the graph record for each step
    #######################################################################################

    ## second method for CPU graph update (fast) ##
    #######################################################################################
    def update_graph(self,start_vertex,end_vertex,graph):
        # Converte in interi
        start_vertex = np.array(start_vertex).round().astype(int)
        end_vertex = np.array(end_vertex).round().astype(int)

        # Estrae le coordinate per la funzione 'draw_line'
        r0, c0 = start_vertex[0], start_vertex[1]
        r1, c1 = end_vertex[0], end_vertex[1]

        # Calcola le coordinate della linea
        rr, cc = draw_line(r0, c0, r1, c1)

        # Filtra le coordinate che sono fuori dai bordi (0-999)
        valid_idx = (rr >= 0) & (rr < 1000) & (cc >= 0) & (cc < 1000)
        rr, cc = rr[valid_idx], cc[valid_idx]

        # Imposta i pixel a 1 sull'array NumPy (super veloce)
        graph[0, 0, rr, cc] = 1.0
    #######################################################################################


    #############################-Expert exploration-######################################

    def expert_restricted_exploration(self,pre_coord,cropped_feature_tensor,orientation_map,correction=False, tolerance = 15): # for training with expert demonstration

        crop_size = self.crop_size # crop size in this case is 63 see above

        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy()) # convert the pre_coord from training coordinate to world coordinate ## ATTENZIONE ###

        # next vertex for updating
        v_next = pre_coord # initialize the next vertex as the pre_coord

        # initialization
        self.agent.taken_stop_action = 0 
        self.agent.gt_stop_action = 0 
        
        # generate the expert demonstration for coord prediction
        if len(self.agent.candidate_label_points): # if there are candidate label points in the cropped region

            l,d,r,u,_,_ = self.agent.crop_info # get the crop info

            # load data
            candidate_label_points = self.agent.candidate_label_points.copy() # candidate label points in the cropped region, 
            candidate_label_points_index = candidate_label_points[:,2] # get the index of the candidate label points
            candidate_label_points = candidate_label_points[:,:2] # get the coordinates of the candidate label points

            # filter points (in this way we are able to handle the situation where the crop region cuts a sequence of candidate points into multiple pieces)
            candidate_label_points_index_1 = candidate_label_points_index[:-1] # index of the candidate label points except the last one
            candidate_label_points_index_2 = candidate_label_points_index[1:] # index of the candidate label points except the first one

            #we want ceck if there are discontinuities in the candidate label points indices
            delta = candidate_label_points_index_2 - candidate_label_points_index_1 # compute the difference between consecutive indices 
            delta = (delta==1) # check if the difference is 1  
            filtered_index = [x for x in range(len(delta)) if delta[x]!=1] # get the indices where the difference is not 1

            # if delta is not 1, it means that a piece of sequence is out of the crop region, so some candidate points are disconnected
            # only save the piece that the current vertex can reach
            if len(filtered_index): # if there are filtered indices
                candidate_label_points = candidate_label_points[:filtered_index[0]+1] # keep only the candidate points up to the first discontinuity 

            

            # generate label

            # first method: nearest neighbor search
            ##########################################
            #tree = cKDTree(candidate_label_points) # create a KDTree for fast nearest neighbor search with the candidate label points 
            #_, iis = tree.query([self.agent.v_now],k=[1]) # connect the current vertex with the nearest candidate label point
            #iis = iis[0][0] # get the index of the nearest candidate label point
            ###########################################

            #second method: Euclidean distance search
            ###########################################
            distances = np.linalg.norm(candidate_label_points[:, :2] - self.agent.v_now, axis=1)
            iis = np.argmin(distances) # get the index of the nearest candidate label point
            ###########################################

            v_center = candidate_label_points[iis] # get the coordinates of the nearest candidate label point
            candidate_label_points = candidate_label_points[iis:] # keep only the candidate points ahead of the current vertex, in this way we avoid going backward
            # 
            cropped_ahead_points = (candidate_label_points[:,0]>=max(d,v_center[0]-15)) * (candidate_label_points[:,0]<min(u,v_center[0]+15)) * \
                            (candidate_label_points[:,1] >=max(l,v_center[1]-15)) * (candidate_label_points[:,1] <min(r,v_center[1]+15)) # filter the candidate points that are within a 15-pixel range from the current vertex in both x and y directions, we obtain a value array where 1 indicates the point is within the range and 0 otherwise
            
            points_index = np.where(cropped_ahead_points==1) # get the indices of the candidate points that are within the range
            candidate_label_points = candidate_label_points[points_index] # keep only the candidate points within the range

            if not len(candidate_label_points): # if there are no candidate label points within the range

                self.agent.gt_stop_action = 1 # set the ground truth stop action to 1
                gt_coord = None # set the ground truth coordinate to None because there are no candidate points

                # after reaching the condition of no candidate points, we also check if we are close to the end vertex
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10): # if the distance between the current vertex and the end vertex is less than 10 pixels
                    self.agent.taken_stop_action = 1 # set the taken stop action to 1
                    gt_coord = self.agent.end_vertex.copy() # set the ground truth coordinate to the end vertex
                    self.agent.candidate_label_points = [gt_coord] # update the candidate label points to include the end vertex
            
            # if there are candidate label points within the range

            #### SOLUTION WHIT ORIENTATION MAP 
            #else:
        
            #    orientation = [orientation_map[int(x[0]),int(x[1])] for x in candidate_label_points] # get the orientation values for the candidate label points from the orientation map
            #    ori_now = orientation_map[int(v_center[0]),int(v_center[1])] # get the orientation value for the current vertex from the orientation map

            #    if ori_now > 1: # if the orientation value is greater than 1 
            #       ori_now_left = ori_now - 1 # get the left orientation value
            #    else:
            #        ori_now_left = 64 
            #    if ori_now != 64:
            #        ori_now_right = ori_now + 1
            #    else:
            #        ori_now_right = 1
                # locate corner pixels (pixels whose orientation is +/- 5 degree)


            #    gt_candiate = np.where((orientation!=ori_now)*(orientation!=ori_now_left)*(orientation!=ori_now_right))[0] # get the indices of the candidate points whose orientation is different from the current orientation and its neighbors,the result is true only when the orientation is different from all three values simultaneously
            #    if len(gt_candiate): # if there are candidate points with different orientation
            #        gt_coord = candidate_label_points[min(len(candidate_label_points)-1,min(gt_candiate)+5)] # select the ground truth coordinate as the candidate point that is 5 positions ahead of the first candidate point with different orientation
            #    else:
            #        gt_coord = candidate_label_points[len(candidate_label_points)-1] # if all candidate points have the same orientation, select the last candidate point as the ground truth coordinate, we take the point that is farthest from the current vertex

            #SOLUTION WITHOUT ORIENTATION MAP 
            ################################# 
            else:
                STEP_SIZE = 3 
                target_index = STEP_SIZE
                target_index = min(target_index, len(candidate_label_points) - 1)
                gt_coord = candidate_label_points[target_index]
            #################################
            
                dd = np.linalg.norm(gt_coord - pre_coord) # compute the distance between the ground truth coordinate and the previous coordinate
                gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1) # get the index of the ground truth coordinate in the instance vertices list 

                if (self.agent.ii <= self.agent.pre_ii): # if the current index is less than or equal to the previous index, it means we are going backward
                    self.agent.gt_stop_action = 1 # set the ground truth stop action to 1
                self.agent.pre_ii = self.agent.ii # update the previous index
                self.agent.ii = gt_index          # update the current index

                if dd > tolerance : # if the distance between the ground truth coordinate and the previous coordinate is greater than 15 pixels, so we are in the case where the expert demonstration is far from the predicted coordinate (we add different tolerance distance)
                    v_next = gt_coord.copy() # set the next vertex to the ground truth coordinate

        # if there are no candidate label points in the cropped region            
        else:
            self.agent.gt_stop_action = 1 # set the ground truth stop action to 1
            # whether reach the end vertex
            gt_coord = None # set the ground truth coordinate to None

            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10): # if the distance between the current vertex and the end vertex is less than 10 pixels
                self.agent.taken_stop_action = 1
                gt_coord = self.agent.end_vertex.copy()
                self.agent.candidate_label_points = [gt_coord]

        if gt_coord is not None: # if the ground truth coordinate is not None

            gt_coord[0] -= self.agent.crop_info[4] # convert the ground truth coordinate to the cropped coordinate system
            gt_coord[1] -= self.agent.crop_info[5] # convert the ground truth coordinate to the cropped coordinate system
            gt_coord = [x/(self.crop_size//2) for x in gt_coord] # normalize the ground truth coordinate to the range [-1,1] based on the crop size

        else:
            gt_coord = [-3,-3] # if the ground truth coordinate is None, set it to [-3,-3] to indicate no valid coordinate

        # update graph on GPU
        ##############################
        #self.update_graph(self.agent.v_now,v_next,self.graph_record) # update the graph record with the current vertex and the next vertex
        ##############################

        # update graph on CPU
        ##############################
        self.update_graph(self.agent.v_now,v_next,self.graph_record_np) 
        ##############################


        # save data
        v_now_save = [x/1000 for x in self.agent.v_now]  # normalize the current vertex to the range [0,1] based on the image size
        v_previous_save = [x/1000 for x in self.agent.v_previous] # normalize the previous vertex to the range [0,1] based on the image size
       
        stored_data =  {
            'cropped_feature_tensor':cropped_feature_tensor.detach().cpu(),
            'gt_coord':gt_coord,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}

        self.update_DAgger_buffer(stored_data)
        
        # update
        self.agent.v_previous = self.agent.v_now
        self.agent.v_now = v_next
        
    ##############################-Expert-free exploration-######################################

    def expert_free_exploration(self,pre_coord,cropped_feature_tensor,orientation_map,correction=False):
        crop_size = self.crop_size
        # coord convert
        pre_coord = self.agent.train2world(pre_coord.cpu().detach().numpy())
        # next vertex for updating
        v_next = pre_coord
        # initialization
        self.agent.taken_stop_action = 0
        self.agent.gt_stop_action = 0
        
        # generate the expert demonstration for coord prediction
        if len(self.agent.candidate_label_points):

            l,d,r,u,_,_ = self.agent.crop_info
            # load data
            candidate_label_points = self.agent.candidate_label_points.copy()
            candidate_label_points_index = candidate_label_points[:,2]
            candidate_label_points = candidate_label_points[:,:2]
            # filter points
            candidate_label_points_index_1 = candidate_label_points_index[:-1]
            candidate_label_points_index_2 = candidate_label_points_index[1:]
            # if only one candidate instances in the cropped region, all elements in delta should be 1
            delta = candidate_label_points_index_2 - candidate_label_points_index_1
            delta = (delta==1)
            filtered_index = [x for x in range(len(delta)) if delta[x]!=1]
            # if delta is not 1, it means a instance is cut into multiple pieces by the crop action
            # only save the piece that the current vertex can reach
            if len(filtered_index):
                candidate_label_points = candidate_label_points[:filtered_index[0]+1]

            #generate label

            # first method: nearest neighbor search
            ##########################################
            #tree = cKDTree(candidate_label_points) # create a KDTree for fast nearest neighbor search with the candidate label points 
            #_, iis = tree.query([self.agent.v_now],k=[1]) # connect the current vertex with the nearest candidate label point
            #iis = iis[0][0] # get the index of the nearest candidate label point
            ###########################################

            #second method: Euclidean distance search
            ###########################################
            distances = np.linalg.norm(candidate_label_points[:, :2] - self.agent.v_now, axis=1)
            iis = np.argmin(distances) # get the index of the nearest candidate label point
            ###########################################            
            

            v_center = candidate_label_points[iis]
            candidate_label_points = candidate_label_points[iis:]

            # 
            cropped_ahead_points = (candidate_label_points[:,0]>=max(d,v_center[0]-15)) * (candidate_label_points[:,0]<min(u,v_center[0]+15)) * \
                            (candidate_label_points[:,1] >=max(l,v_center[1]-15)) * (candidate_label_points[:,1] <min(r,v_center[1]+15))
            points_index = np.where(cropped_ahead_points==1)
            candidate_label_points = candidate_label_points[points_index]

            if not len(candidate_label_points):
                self.agent.gt_stop_action = 1
                gt_coord = None
                if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                    self.agent.taken_stop_action = 1
                    gt_coord = self.agent.end_vertex.copy()
                    self.agent.candidate_label_points = [gt_coord]

            # SOLUTION WHIT ORIENTATION MAP 
            #else:
            #    orientation = [orientation_map[int(x[0]),int(x[1])] for x in candidate_label_points]
            #    ori_now = orientation_map[int(v_center[0]),int(v_center[1])]
            #    if ori_now > 1:
            #        ori_now_left = ori_now - 1
            #    else:
            #        ori_now_left = 64
            #    if ori_now != 64:
            #        ori_now_right = ori_now + 1
            #    else:
            #        ori_now_right = 1
                # locate corner pixels (pixels whose orientation is +/- 5 degree)
            #    gt_candiate = np.where((orientation!=ori_now)*(orientation!=ori_now_left)*(orientation!=ori_now_right))[0]
            #    if len(gt_candiate):
            #        gt_coord = candidate_label_points[min(len(candidate_label_points)-1,min(gt_candiate)+5)]
            #    else:
            #        gt_coord = candidate_label_points[len(candidate_label_points)-1]

            #SOLUTION WITHOUT ORIENTATIONMAP
            ###############################
            else: 
                STEP_SIZE = 3
                target_index = STEP_SIZE
                target_index = min(target_index, len(candidate_label_points) - 1)
                gt_coord = candidate_label_points[target_index]
            ###############################
                
                # dd = np.linalg.norm(gt_coord - pre_coord)
                gt_index = next((i for i, val in enumerate(self.agent.instance_vertices) if np.all(val==gt_coord)), -1)
                if (self.agent.ii <= self.agent.pre_ii):
                    self.agent.gt_stop_action = 1
                self.agent.pre_ii = self.agent.ii
                self.agent.ii = gt_index  


                # correction mechanism only for the first epoch to avoid drifting during training 
                if correction and self.epoch_counter==1:
                    beta = 0.5**(self.training_step/1000)
                    v_next = (np.array(v_next) * (1 - beta) + np.array(gt_coord) * beta)
                    v_next = [int(v_next[0]),int(v_next[1])]

        # if there are no candidate label points in the cropped region        
        else:
            self.agent.gt_stop_action = 1
            # whether reach the end vertex
            gt_coord = None
            if (np.linalg.norm(np.array(self.agent.v_now) - np.array(self.agent.end_vertex))<10):
                self.agent.taken_stop_action = 1
                gt_coord = self.agent.end_vertex.copy()
                self.agent.candidate_label_points = [gt_coord]
        if gt_coord is not None:
            gt_coord[0] -= self.agent.crop_info[4]
            gt_coord[1] -= self.agent.crop_info[5]
            gt_coord = [x/(self.crop_size//2) for x in gt_coord]
        else:
            gt_coord = [-3,-3]

        # update graph on GPU
        ##############################
        #self.update_graph(self.agent.v_now,v_next,self.graph_record) # update the graph record with the current vertex and the next vertex
        ##############################

        # update graph on CPU
        ##############################
        self.update_graph(self.agent.v_now,v_next,self.graph_record_np) 
        ##############################       
       

        # save data
        v_now_save = [x/1000 for x in self.agent.v_now]
        v_previous_save = [x/1000 for x in self.agent.v_previous]
        stored_data =  {
            'cropped_feature_tensor':cropped_feature_tensor.detach().cpu(),
            'gt_coord':gt_coord,
            'gt_stop_action':self.agent.gt_stop_action,
            'v_now':v_now_save,
            'v_previous':v_previous_save}

        self.update_DAgger_buffer(stored_data)
        
        # update
        self.agent.v_previous = self.agent.v_now
        self.agent.v_now = v_next

    def remove_duplicate_init_points(self,v):
        for u in self.init_point_set:
            dis = np.linalg.norm(np.array(u)-np.array(v))
            if dis < 10:
                self.init_point_set.remove(u)

class Agent(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # state
        self.v_now = [0,0]
        self.v_previous = [0,0]
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        #
        self.instance_vertices = np.array([])
        self.candidate_label_points = np.array([])
        self.tree = None
        self.crop_info = []
        self.init_vertex = [0,0]
        self.end_vertex = [0,0]
        #
        self.ii = 0
        self.pre_ii = -1
        self._freeze()

    def init_agent(self,init_vertex):
        self.taken_stop_action = 0
        self.gt_stop_action = 0
        self.agent_step_counter = 0
        self.v_now = init_vertex
        self.v_previous = init_vertex
        self.ii = 0
        self.pre_ii = -1
        self.local_loop_repeat_counter = 0
        self.empty_crop_repeat_counter = 0
        
    def train2world(self,coord_in,crop_info=None):
        if crop_info is None:
            crop_info = self.crop_info    
        crop_size = self.env.crop_size
        pre_coord = [int(x*(crop_size//2)) for x in coord_in]
        pre_coord[0] += crop_info[4] 
        pre_coord[1] += crop_info[5]
        pre_coord = [max(min(pre_coord[0],crop_info[3]-1),crop_info[1]),max(min(pre_coord[1],crop_info[2]-1),crop_info[0])]
        return pre_coord

    def crop_attention_region(self,fpn_feature_tensor,val_flag=False):
        r'''
            Crop the current attension region centering at v_now.
        '''
        crop_size = self.env.crop_size
        # find left, right, up and down positions
        l = self.v_now[1]-crop_size//2
        r = self.v_now[1]+crop_size//2+1
        d = self.v_now[0]-crop_size//2
        u = self.v_now[0]+crop_size//2+1
        crop_l, crop_r, crop_d, crop_u = 0, self.env.crop_size, 0, self.env.crop_size
        if l<0:
            crop_l = -l
        if d<0:
            crop_d = -d
        if r>1000:
            crop_r = crop_r-r+1000
        if u>1000:
            crop_u = crop_u-u+1000
        crop_l,crop_r,crop_u,crop_d = int(crop_l),int(crop_r),int(crop_u),int(crop_d)
        l,r,u,d = max(0,min(1000,int(l))),max(0,min(1000,int(r))),max(0,min(1000,int(u))),max(0,min(1000,int(d)))
        self.crop_info = [l,d,r,u,self.v_now[0],self.v_now[1]]

        ## sincronize the graph record from CPU to GPU (slow) ##
        ##########################################
        # cropped feature tensor for iCurb
        #cropped_feature_tensor = torch.zeros(1,8,self.env.crop_size,self.env.crop_size)
        #cropped_graph = torch.zeros(1,1,self.env.crop_size,self.env.crop_size)
        #cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r]
        #cropped_graph[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record[:,:,d:u,l:r]
        #cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach()
        ##########################################

        ## sincronize the graph record from CPU to GPU (fast) ##
        ##########################################
        # cropped feature tensor for iCurb
        
        cropped_feature_tensor = torch.zeros(1,8,self.env.crop_size,self.env.crop_size).to(self.args.device)
        cropped_feature_tensor[:,:,crop_d:crop_u,crop_l:crop_r] = fpn_feature_tensor[:,:,d:u,l:r]

        cropped_graph_np = np.zeros((1, 1, self.env.crop_size, self.env.crop_size), dtype=np.float32)
        cropped_graph_np[:,:,crop_d:crop_u,crop_l:crop_r] = self.env.graph_record_np[:,:,d:u,l:r]

        cropped_graph = torch.from_numpy(cropped_graph_np).to(self.args.device)

        cropped_feature_tensor = torch.cat([cropped_feature_tensor,cropped_graph],dim=1).detach()
        ##########################################

        # update the gt pixels within the cropped region
        if not val_flag:
            ahead_points = self.instance_vertices[self.ii:]
            ahead_points = np.array([[x[0],x[1],i] for i,x in enumerate(ahead_points)])
            cropped_ahead_points = (ahead_points[:,0]>=d) * (ahead_points[:,0]<u) * (ahead_points[:,1] >=l) * (ahead_points[:,1] <r)
            points_index = np.where(cropped_ahead_points==1)
            cropped_ahead_points = ahead_points[points_index]
            self.candidate_label_points = cropped_ahead_points#[x for x in cropped_ahead_points if (((x[0] - self.v_now[0])**2 + (x[1]- self.v_now[1])**2)**0.5>15)]
            # if len(self.candidate_label_points):
            #     self.tree = cKDTree(self.candidate_label_points)
            # else:W
            #     self.tree = None
        return cropped_feature_tensor.to(self.args.device)
    
    

class Network(FrozenClass):
    def __init__(self,env):
        self.env = env
        self.args = env.args
        # initialization
        #self.encoder = FPN()
        self.encoder = FPN(backbone_name='efficientnet_b4', n_channels=4)

        #self.decoder_coord = DecoderCoord(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4)
        #self.decoder_stop = DecoderStop(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4)
        self.decoder_coombined = CombinedDecoder(visual_size=pow(math.ceil(self.env.crop_size/8),2)*32+4)
        self.encoder.to(device=self.args.device)
        #self.decoder_coord.to(device=self.args.device)
        #self.decoder_stop.to(device=self.args.device)
        self.decoder_coombined.to(device=self.args.device)
        # tensorboard
        if not self.args.test:
            #self.writer = SummaryWriter('./records/tensorboard') ## old version ##

            # new version with experiment name
            exp_name = self.args.experiment_name
            log_dir = os.path.join('./records/tensorboard', exp_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # ====================optimizer=======================
        self.optimizer_enc = optim.Adam(list(self.encoder.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        #self.optimizer_coord_dec = optim.Adam(list(self.decoder_coord.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        #self.optimizer_flag_dec = optim.Adam(list(self.decoder_stop.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        self.optimizer_decoder = optim.Adam(list(self.decoder_coombined.parameters()), lr=self.args.lr_rate, weight_decay=self.args.weight_decay)
        # =====================init losses=======================
        criterion_l1 = L1Loss(reduction='mean')
        criterion_bce = BCEWithLogitsLoss()
        criterion_ce = CrossEntropyLoss()
        self.criterions = {"ce":criterion_ce,'l1':criterion_l1,"bce": criterion_bce}
        # =====================Load data========================
        dataset_train = DatasetiCurb(self.args,mode="train")
        dataset_valid = DatasetiCurb(self.args,mode="valid")
        self.dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True,collate_fn=self.iCurb_collate, num_workers = 16, pin_memory = True)
        self.dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False,collate_fn=self.iCurb_collate, num_workers = 16, pin_memory = True)
        print("Dataset splits -> Train: {} | Valid: {}\n".format(len(dataset_train), len(dataset_valid)))
        self.loss = 0
        self.best_f1 = 0
        #
        self.load_checkpoints()
        self._freeze()
    ##load checpoints not combined decoder ##
    ##########################################
    #def load_checkpoints(self):
    #    device = self.args.device
    #   self.encoder.load_state_dict(torch.load('./checkpoints/seg_pretrain.pth',map_location= device))
    #    if self.args.test:
    #        self.decoder_coord.load_state_dict(torch.load("./checkpoints/decoder_nodis_coord_best.pth",map_location= device))
    #        self.decoder_stop.load_state_dict(torch.load("./checkpoints/decoder_nodis_flag_best.pth",map_location=device))
    #        print('=============')
    #        print('Successfully loading iCurb checkpoints!')
        
    #    # self.decoder_seg.load_state_dict(torch.load('./dataset/pretrain_mask_decoder_19.pth',map_location='cpu'))
    #    print('=============')
    #    print('Pretrained FPN encoder checkpoint loaded!')
    ##########################################


    ##load checpoints combined decoder ##
    ##########################################
    def load_checkpoints(self):
        device = self.args.device
        self.encoder.load_state_dict(torch.load('./checkpoints/seg_pretrain_manhattan_efficentnet_1.6.pth',map_location= device))
        print('=============')
        print('Pretrained FPN encoder checkpoint loaded!')

        # Non carichiamo i checkpoint se non siamo in modalità test O se stiamo continuando un training
        # NOTA: se vuoi ri-addestrare da zero, commenta il blocco 'if'
        if self.args.test: # O se hai un flag 'resume_training'
            print('=============')
            print('Fusing old checkpoints into new CombinedDecoder...')
            try:
                # Carica i vecchi pesi
                coord_path = "./checkpoints/decoder_nodis_coord_best.pth"
                stop_path = "./checkpoints/decoder_nodis_flag_best.pth"
                coord_weights = torch.load(coord_path, map_location=device)
                stop_weights = torch.load(stop_path, map_location=device)

                # Prendi il dizionario del nuovo modello
                new_state_dict = self.decoder_coombined.state_dict()

                # 1. Copia i pesi condivisi e la testa "coord" dal modello 'coord'
                for k, v in coord_weights.items():
                    if 'linear1' in k: # Rinomina linear1 -> linear_shared
                        new_k = k.replace('linear1', 'linear_shared')
                    elif 'linear2' in k: # Rinomina linear2 -> head_coord
                        new_k = k.replace('linear2', 'head_coord')
                    else:
                        new_k = k # Tutti gli altri (conv, res_layer) hanno lo stesso nome

                    if new_k in new_state_dict:
                        new_state_dict[new_k] = v

                # 2. Copia SOLO la "testa" dal modello 'stop'
                new_state_dict['head_stop.weight'] = stop_weights['linear2.weight']
                new_state_dict['head_stop.bias'] = stop_weights['linear2.bias']

                # Carica il dizionario fuso
                self.decoder_coombined.load_state_dict(new_state_dict)
                print('Successfully loaded and fused iCurb checkpoints!')

            except Exception as e:
                print(f"Error loading checkpoints: {e}")
                print("Could not merge old checkpoints. Loading stopped.")

    ##########################################
       
    def train_mode(self):
        #self.decoder_coord.train()
        #self.decoder_stop.train()
        self.decoder_coombined.train()
    
    def val_mode(self):
        #self.decoder_coord.eval()
        #self.decoder_stop.eval()
        self.decoder_coombined.eval()
    
    def train_len(self):
        return len(self.dataloader_train)

    def val_len(self):
        return len(self.dataloader_valid)

    def bp(self):
        #self.optimizer_coord_dec.zero_grad()
        #self.optimizer_flag_dec.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.loss.backward()
        #self.optimizer_flag_dec.step()
        #self.optimizer_coord_dec.step()
        self.optimizer_decoder.step()
        self.loss = 0

    def save_checkpoints(self,i):
        print('Saving checkpoints {}.....'.format(i))
        #torch.save(self.decoder_coord.state_dict(), "./checkpoints/decoder_nodis_coord_best.pth")
        #torch.save(self.decoder_stop.state_dict(), "./checkpoints/decoder_nodis_flag_best.pth")

        #default saving for combined decoder
        #torch.save(self.decoder_coombined.state_dict(), "./checkpoints/decoder_combined_best.pth")
        
        # saving with experiment name
        exp_name = self.args.experiment_name
        save_path = os.path.join("./checkpoints", f"decoder_coombined_best_{exp_name}.pth")
        torch.save(self.decoder_coombined.state_dict(), save_path)


    def DAgger_collate(self,batch):
        # variables as tensor
        cat_tiff = torch.stack([x[0] for x in batch])
        v_now = torch.stack([x[1] for x in batch])
        v_previous = torch.stack([x[2] for x in batch])
        gt_coord = torch.stack([x[3] for x in batch])
        gt_stop_action = torch.stack([x[-1] for x in batch]).reshape(-1)
        
        return cat_tiff, v_now, v_previous, gt_coord, gt_stop_action

    def iCurb_collate(self,batch):
        # variables as numpy
        seq = np.array([x[0] for x in batch])
        mask = np.array([x[3] for x in batch])
        orientation_map = np.array([x[4] for x in batch])
        # variables as list
        seq_lens = [x[1] for x in batch]
        image_name = [x[5] for x in batch]
        init_points = [x[6] for x in batch]
        end_points = [x[7] for x in batch]
        # variables as tensor
        tiff = torch.stack([x[2] for x in batch])
        return seq, seq_lens, tiff, mask, orientation_map, image_name, init_points, end_points

