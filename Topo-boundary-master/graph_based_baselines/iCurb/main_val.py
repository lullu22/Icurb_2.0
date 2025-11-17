import numpy as np
import torch
import random
from PIL import Image
from scipy.spatial import cKDTree
import torch.nn.functional as F
import os
import scipy
import json

def run_val(env,val_index): # validate on the whole validation set
    agent = env.agent # get the agent, this means the iCurb path planner
    network = env.network # get the network, this means the iCurb network
    args = env.args 
    crop_size = env.crop_size # get the crop size


    def update_coord(pre_coord): # update the vertex coord with the predicted offset
        pre_coord = pre_coord.cpu().detach().numpy() # get the predicted offset from the network
        v_next = agent.train2world(pre_coord) # convert from prediction value to image coordinate system
        if (agent.v_now == v_next): 
            agent.taken_stop_action = 1 # if the predicted coord is the same as the current coord, take the stop action
        agent.v_now = v_next 
        
    def update_graph(start_vertex,end_vertex,graph,set_value=1): # update the graph with the new edge
        start_vertex = np.array([int(start_vertex[0]),int(start_vertex[1])]) 
        end_vertex = np.array([int(end_vertex[0]),int(end_vertex[1])])
        instance_vertices = [] 
        p = start_vertex # current position
        d = end_vertex - start_vertex # direction vector
        N = np.max(np.abs(d))  # number of steps to take
        graph[:,:,start_vertex[0],start_vertex[1]] = set_value # set the start vertex
        graph[:,:,end_vertex[0],end_vertex[1]] = set_value # set the end vertex
        if N: # if the number of steps is greater than 0
            s = d / (N) # step size
            for i in range(0,N):
                p = p + s # update the current position 
                graph[:,:,int(round(p[0])),int(round(p[1]))] = set_value # set the vertex
        return graph # return the updated graph

    def visualization_image(graph_record,name,mask): # visualize the graph on the image and save it
        output = graph_record[0,0,:,:].cpu().detach().numpy()
        points = np.where(output!=0)   
        if len(points[0]): #
            if args.test: # if in testing mode, save the visualization results in the test folder
                graph_map = Image.fromarray(output * 255).convert('RGB').save('./records/test/skeleton/{}.png'.format(name))  #  only for test
            else:    
                graph_map = Image.fromarray(output * 255).convert('RGB').save('./records/valid/vis/{}_{}.png'.format(val_index,name)) # save the visualization results in the validation folder

    def eval_metric(graph,mask,name): # evaluate the graph with the ground truth mask
        def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))] # convert tuple to list, because we need to use the list to build the KDTree

        graph = graph.cpu().detach().numpy() # convert the graph to numpy array
        gt_image = mask # get the ground truth mask
        gt_points = tuple2list(np.where(gt_image!=0)) # get the ground truth points
        graph_points = tuple2list(np.where(graph!=0)) # get the graph points

        graph_acc = 0 # accuracy
        graph_recall = 0 # recall
        gt_tree = scipy.spatial.cKDTree(gt_points) # build the KDTree for the ground truth points
        for c_i,thre in enumerate([5]): 
            if len(graph_points): # if there are graph points
                graph_tree = scipy.spatial.cKDTree(graph_points) # build the KDTree for the graph points
                dis_gt2graph,_ = graph_tree.query(gt_points, k=1) # query the nearest graph point for each ground truth point
                dis_graph2gt,_ = gt_tree.query(graph_points, k=1) # query the nearest ground truth point for each graph point
                graph_recall = len([x for x in dis_gt2graph if x<thre])/len(dis_gt2graph) # calculate the recall
                graph_acc = len([x for x in dis_graph2gt if x<thre])/len(dis_graph2gt) # calculate the accuracy
        if graph_acc*graph_recall: 
            r_f = 2*graph_recall * graph_acc / (graph_acc+graph_recall) # calculate the F1 score
        return graph_acc, graph_recall, r_f # return the accuracy, recall and F1 score


    network.val_mode() # set the network to validation mode
    eval_len = network.val_len() # get the length of the validation set
    graph_acc_ave = 0 
    graph_recall_ave=0
    r_f_ave = 0
    # =================working on an image=====================
    for i, data in enumerate(network.dataloader_valid): # iterate through the validation set
        _, _, tiff, mask, name, init_points, end_points = data # get the data
        name = name[0][-14:-5]
        tiff = tiff.to(args.device) # change device to GPU
        mask, init_points, end_points = mask[0], init_points[0], end_points[0] # get the data from the list
        init_points = [[int(x[0]),int(x[1])] for x in init_points] # convert the init points to int
        
        
        # clear history info
        env.init_image(valid=True) # initialize the environment for validation

        with torch.no_grad():# no gradient calculation, we only need the forward pass, method provided by PyTorch 
          
            _, fpn_feature_map = network.encoder(tiff) # extract the feature map from the tiff image using the encoder

            # pre_seg_mask = network.decoder_seg(fpn_feature_map)
            fpn_feature_map = F.interpolate(fpn_feature_map, size=(1000, 1000), mode='bilinear', align_corners=True) # upsample the feature map to the original image size
            fpn_feature_tensor = fpn_feature_map#torch.cat([tiff,fpn_feature_map],dim=1) 
            # mask_to_save = torch.sigmoid(pre_seg_mask).squeeze(0).squeeze(0).cpu().detach().numpy()
            # output_img = mask_to_save
            # save_img = Image.fromarray( (output_img/np.max(output_img) )* 255) 
            # save_img.convert('RGB').save('./records/output_seg/{}.png'.format(name))

        # record generated vertices
        vertices_record_list = []
        if len(init_points): #
            for v_idx,init_point in enumerate(init_points): # iterate through the init points where v_idx is the index of the init point and init_point is the coord of the init point
                # ===============working on a curb instance======================
                agent.init_agent(init_point) 
                vertices_record_list.append(init_point) 

                while 1: # keep growing the graph until the stop action is taken

                    agent.agent_step_counter += 1 # count the steps taken by the agent

                    # network predictions
                    cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor,val_flag=True) # crop the attention region around the current vertex

                    with torch.no_grad(): # no gradient calculation, we only need the forward pass, method provided by PyTorch
                        v_now = [x/1000 for x in agent.v_now] # normalize the current vertex coord to [0,1]
                        v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device) # convert to tensor and change device to GPU

                        v_previous = [x/1000 for x in agent.v_previous] # normalize the previous vertex coord to [0,1]
                        v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device) # convert to tensor and change device to GPU

                        pre_stop_action = network.decoder_stop(cropped_feature_tensor,v_now,v_previous) # predict the stop action
                        pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous) # predict the coord offset

                    pre_stop_action = pre_stop_action.squeeze(1) # get the stop action prediction from the network and squeeze the tensor
                    pre_coord = pre_coord.squeeze(0).squeeze(0) # get the coord prediction from the network and squeeze the tensor

                    pre_stop_action = F.softmax(pre_stop_action,dim=1) # apply softmax to the stop action prediction to get the probabilities
                    agent.v_previous = agent.v_now # in this step, the current vertex becomes the previous vertex

                    # update vertex coord
                    update_coord(pre_coord) # update the current vertex coord with the predicted offset

                    # take stop action
                    if (pre_stop_action[0][1] > 0.5 and agent.agent_step_counter > 20): # if the probability of stop action is greater than 0.5 and the agent has taken more than 20 steps, take the stop action
                        break

                    # record
                    env.graph_record = update_graph(agent.v_now,agent.v_previous,env.graph_record) # update the graph with the new edge
                    vertices_record_list.append(np.array(agent.v_now).tolist()) # record the new vertex
                    if (agent.agent_step_counter > args.max_length*1.2) \
                        or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0) or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10): # if the agent has taken more than max_length*1.2 steps or the agent has reached the image boundary and taken more than 10 steps, take the stop action
                        agent.taken_stop_action = 1
                        
                    if agent.taken_stop_action:    
                        break
                        
        # calculate metrics
        graph_acc, graph_recall, r_f = eval_metric(env.graph_record[0,0],mask,name) # compute the evaluation metrics accuracy, recall and F1 score
        # average the metrics
        graph_acc_ave = (graph_acc_ave * i + graph_acc) / (i + 1)
        graph_recall_ave = (graph_recall_ave * i + graph_recall) / (i + 1)
        r_f_ave = (r_f_ave * i + r_f) / (i + 1)

        # vis and print
        visualization_image(env.graph_record,name,mask) # visualize the graph on the image and save it
        print('Validation {}-{}/{} || graph {}/{}/{}'.format(name,i,eval_len,round(graph_acc,4)
                ,round(graph_recall,4),round(r_f_ave,4))) 
        

        ######################## recording ########################
        
        # save metrics in txt if in testing mode only for the first 100 images

        # create the directory if it does not exist
        if args.test and not os.path.exists('./records/test/metrics'):
            os.makedirs('./records/test/metrics')
        if args.test and i<100:
            with open('./records/test/metrics/metrics.txt','a') as f:
                f.write('{}-{}/{} || graph {}/{}/{}\n'.format(name,i,eval_len,round(graph_acc,4)
                    ,round(graph_recall,4),round(r_f_ave,4)))
        
        ###########################################################
      
            


        ###########################################################

        
        # save the recorded vertices
        if args.test: # if in testing mode, save the recorded vertices in the test folder
            with open('./records/test/vertices_record/{}.json'.format(name),'w') as jf:
                json.dump(vertices_record_list,jf) 
        else: # if in validation mode, save the recorded vertices in the validation folder
            with open('./records/valid/vertices_record/{}.json'.format(name),'w') as jf:
                json.dump(vertices_record_list,jf)
    # finish
    print('----Finishing Validation || graph {}/{}'.format(round(graph_acc_ave,4),round(graph_recall_ave,4))) 
    if not args.test:   
        network.writer.add_scalars('Val/Val_Accuracy by image',{'graph_acc':graph_acc_ave},env.training_step)
        network.writer.add_scalars('Val/Val_Recall by image',{'graph_recall':graph_recall_ave},env.training_step)
        network.writer.add_scalars('Val/Val_F1 by image',{'graph_f1':r_f_ave},env.training_step)
        
    return r_f_ave

