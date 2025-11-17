import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from main_sampler import  run_free_explore, run_restricted_explore
from utils.dataset import DatasetiCurb, DatasetDagger
import torch.nn.functional as F
from scipy.spatial import cKDTree

def train_Dagger(env,data,index):
    r'''
        Train iCurb with batch-data in DAgger buffer
    '''
    crop_size = env.crop_size
    train_len = env.network.train_len()
    # load data
    cropped_feature_tensor, batch_v_now, batch_v_previous, gt_coords, gt_stop_actions = data
    # make prediction
    cropped_feature_tensor = cropped_feature_tensor.to(env.args.device).squeeze(1)

    # uncomment these lines if we want to use separate decoders for coord and stop
    ##########################################################
    #pre_stop_actions = env.network.decoder_stop(cropped_feature_tensor, torch.FloatTensor(batch_v_now).to(env.args.device),torch.FloatTensor(batch_v_previous).to(env.args.device))
    #pre_stop_actions = pre_stop_actions.squeeze(1)
    #pre_coords = env.network.decoder_coord(cropped_feature_tensor, torch.FloatTensor(batch_v_now).to(env.args.device),torch.FloatTensor(batch_v_previous).to(env.args.device))
    ##########################################################

    # comment this line if we want back to separate decoders for coord and stop
    ##########################################################
    pre_stop_actions, pre_coords = env.network.decoder_coombined(cropped_feature_tensor,torch.FloatTensor(batch_v_now).to(env.args.device),torch.FloatTensor(batch_v_previous).to(env.args.device))
    pre_stop_actions = pre_stop_actions.squeeze(1)
    ##########################################################
    
    # generating gt labels for coord prediction
    gt_coords_train = []
    pre_coords_train = []
    for i,gt_coord in enumerate(gt_coords): 
        if gt_coord[0,0]!=-3:
            pre_coords_train.append(pre_coords[i])
            gt_coords_train.append(gt_coord)

    # loss
    gt_stop_actions = torch.LongTensor(gt_stop_actions).to(env.args.device)
    loss_stop = env.network.criterions['ce'](pre_stop_actions,gt_stop_actions)
    if len(pre_coords_train):
        gt_coords = torch.stack(gt_coords_train).to(env.args.device).squeeze(1)
        pre_coords = torch.stack(pre_coords_train).to(env.args.device)
        loss_coord = env.network.criterions['l1'](pre_coords,gt_coords)
        loss = loss_stop + loss_coord 
        env.network.loss = loss
        env.network.bp()
        return loss_coord.item(), loss_stop.item()
    else:
        loss = loss_stop 
        env.network.loss = loss
        env.network.bp()
        return 0, loss_stop.item()
    
def run_train(env,data,iCurb_image_index):
    def train():
        r'''
            Train iCurb with the aggregated DAgger buffer
        '''
        loss_coord_ave = 0
        loss_stop_ave = 0
        dataset = DatasetDagger(env.DAgger_buffer)
        data_loader = DataLoader(dataset, batch_size=env.args.batch_size, shuffle=True,collate_fn=env.network.DAgger_collate)
        if len(dataset):
            for ii, data_explore in enumerate(data_loader):
                loss_coord, loss_stop = train_Dagger(env,data_explore,iCurb_image_index)
                loss_coord_ave = (loss_coord_ave * ii + loss_coord) / (ii + 1)
                loss_stop_ave = (loss_stop_ave * ii + loss_stop) / (ii + 1)
        return loss_coord_ave, loss_stop_ave

    env.training_step += 1
    train_len = env.network.train_len()
    # get tiff image data
    seq, seq_lens, tiff, mask, orientation_map, name, init_points, end_points = data
    tiff = tiff.to(env.args.device)
    seq, seq_lens, mask, init_points, end_points = seq[0], seq_lens[0], mask[0], init_points[0], end_points[0]
    name = name[0][-14:-5]
    orientation_map = orientation_map[0]

    #comment these lines if we want use the pre-computed features (with ##)
    ########################################################################
    # extract feature of the whole image to grow the graph 
    ##fpn_feature_map = env.network.encoder(tiff)
    # fpn_segmentation_map = network.decoder_seg(fpn_feature_map)



    #fpn_feature_map = F.interpolate(fpn_feature_map, size=(1000, 1000), mode='bilinear', align_corners=True)  
    ##fpn_feature_map = F.interpolate(fpn_feature_map, scale_factor = 4, mode='bilinear', align_corners=True)  



    ##fpn_feature_tensor = fpn_feature_map #torch.cat([tiff,fpn_feature_map],dim=1)
    #########################################################################

    # uncomment this line if we want use the pre-computed features (with ##)
    #########################################################################
    fpn_feature_tensor = tiff
    #########################################################################

    restrictive_tolerance = 5 # value for the second restricted exploration 
    num_free_exploration = 5 # number of free exploration 

    env.DAgger_buffer = [] 
    print(f'--------- Buffer DAgger cleaned for the new image ---------')

    # 1. Restricted Exploration (Tolleranza 15)
    print(f'--------- Running Restricted Exploration (15px Tol) ---------')
    run_restricted_explore(env, seq, fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points, iCurb_image_index, 1, args = env.args, tolerance=15 )
    
    #first training on first restricted exploration 
    if len(env.DAgger_buffer) >= env.args.batch_size:
        print(f'--- Training su D (size: {len(env.DAgger_buffer)}) ---')
        loss_coord, loss_stop = train()
        env.last_loss_coord = loss_coord
        env.last_loss_stop = loss_stop
    else:
        loss_coord = env.last_loss_coord
        loss_stop = env.last_loss_stop

    # 2. Restricted Exploration (Tolleranza 5)
    print(f'--------- Running Restricted Exploration (restrictive_tolerance) ---------')
    run_restricted_explore(env, seq, fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points, iCurb_image_index, 2, args = env.args, tolerance=restrictive_tolerance )
    
    #first training on first restricted exploration 
    if len(env.DAgger_buffer) >= env.args.batch_size:
        print(f'--- Training su D (size: {len(env.DAgger_buffer)}) ---')
        loss_coord, loss_stop = train()
        env.last_loss_coord = loss_coord
        env.last_loss_stop = loss_stop
    else:
        loss_coord = env.last_loss_coord
        loss_stop = env.last_loss_stop

    # 3. loop for free exploration (num_free_exploration)
    print(f'--------- Running {num_free_exploration} Free Explorations ---------')
    for i in range(num_free_exploration):
        print(f'--- Running Free Exploration {i+1}/{num_free_exploration} ---')
        run_free_explore(env, seq, fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points, iCurb_image_index, i+1, args=env.args)

        if len(env.DAgger_buffer) >= env.args.batch_size:
            print(f'--- Training su D (size: {len(env.DAgger_buffer)}) ---')
            loss_coord, loss_stop = train()
            env.last_loss_coord = loss_coord
            env.last_loss_stop = loss_stop
        else:
            loss_coord = env.last_loss_coord
            loss_stop = env.last_loss_stop       

    loss_coord = env.last_loss_coord
    loss_stop = env.last_loss_stop  

    ####  old version ####     
    """
    # running sampler and training
    print('--------- Training Enhanced-iCurb ---------')
    run_restricted_explore(env, seq, fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points, iCurb_image_index, 1, args = env.args)

    #### modifications for dagger training ####
    ###########################################
    #train()
    ###########################################
    ###########################################

    # free exploration
    run_free_explore(env, seq, fpn_feature_tensor, seq_lens, mask, orientation_map, name, init_points, end_points, iCurb_image_index, 1, args = env.args)
    
    #modifications for dagger training ###
    ###########################################
    # uncomment these lines if we want to use the DAgger training here
    #loss_coord, loss_stop = train()

    # comment this line if we want back to original DAgger training
    loss_coord = env.last_loss_coord
    loss_stop = env.last_loss_stop

    #we set the frequency of Dagger Training 
    #train_frequency = 10 # every 10 images we do one DAgger training

    #if (iCurb_image_index % train_frequency == 0) and len(env.DAgger_buffer) >= env.args.batch_size:
    if  len(env.DAgger_buffer) >= env.args.batch_size:
        print(f'--------- Training on DAgger Buffer (size: {len(env.DAgger_buffer)}) ---------')
        # compute DAgger training and update the last losses
        loss_coord, loss_stop = train()
        env.last_loss_coord = loss_coord
        env.last_loss_stop = loss_stop

    """
    ##### old version #####


    ###########################################
    ###########################################

    
    # # clear the DAgger buffer
    # env.DAgger_buffer = []

    ###########################################
    ###########################################
    
    # time usage
    time_now = time.time()
    time_used = time_now - env.time_start
    time_remained = time_used / env.training_step * (env.training_image_number - env.training_step)
    speed = time_used / env.training_step
    # speed, used time, remained time
    print('Time usage: Speed {}s/im || Ut {}h || Rt {}h'.format(round(speed,2)
                ,round(time_used/3600,2),round(time_remained/3600,2)))

    # print and training curve
    print('Epoch: {}/{} | Image: {}/{} | Loss: coord {}/stop {}'.format(
            env.epoch_counter,env.args.epochs,iCurb_image_index,train_len,
            round(loss_coord,3),round(loss_stop,3)))
    env.network.writer.add_scalar('Train/coord_loss',loss_coord,env.training_step)
    env.network.writer.add_scalar('Train/stop_loss',loss_stop,env.training_step)

