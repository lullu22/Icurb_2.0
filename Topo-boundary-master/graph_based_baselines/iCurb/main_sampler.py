import numpy as np
import torch
import random
from PIL import Image
from scipy.spatial import cKDTree
import torch.nn.functional as F

def run_restricted_explore(env, seq, fpn_feature_tensor, seq_lens, mask,name, init_points, end_points, iCurb_image_index): # 
    r'''
        The function to run restricted exploration and add generated data samples
        into the Dagger buffer for later training.
    '''

    agent = env.agent
    network = env.network
    args = env.args

    def visualization_image(graph_record,epoch,iCurb_image_index,name,mask):
        graph_record = Image.fromarray(graph_record[0,0,:,:].cpu().detach().numpy() * 255).convert('RGB')
        mask = Image.fromarray(mask * 255).convert('RGB')
        dst = Image.new('RGB', (graph_record.width * 2, graph_record.height ))
        dst.paste(graph_record,(0,0))
        dst.paste(mask,(mask.width,0))
        dst.save('./records/train/vis/restricted_exploration/{}_{}_{}.png'.format(epoch,iCurb_image_index,name))


    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]


    # init environment
    env.init_image() # initialize the environment for a new image
    for instance_id in range(instance_num): # iterate through all curb instances in the image

        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id] # get the vertices of the current curb instance
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy() # set the vertices of the current curb instance to the agent

        if len(agent.instance_vertices): # if the current curb instance is not empty
            init_vertex = init_points[instance_id] # get the initial vertex of the current curb instance
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex) # initialize the agent with the initial vertex
            agent.end_vertex = end_points[instance_id] # set the end vertex of the agent

            while 1:
                agent.agent_step_counter += 1
                # crop rectangle centering v_now
                cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor) # crop the feature map around the current vertex

                with torch.no_grad(): # no gradient calculation

                    v_now = [x/1000 for x in agent.v_now] # normalize the current vertex
                    v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device) # convert to tensor and move to device

                    v_previous = [x/1000 for x in agent.v_previous] # normalize the previous vertex
                    v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device) # convert to tensor and move to device

                    pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous) # predict the next coordinate
                    pre_coord = pre_coord.squeeze(0).squeeze(0) # remove unnecessary dimensions

                env.expert_restricted_exploration(pre_coord,cropped_feature_tensor=cropped_feature_tensor) # run the expert policy for restricted exploration

                # stop action
                if agent.agent_step_counter >= min(seq_lens[instance_id],1000): # if the agent has reached the maximum number of steps
                    agent.taken_stop_action = 1
                if agent.taken_stop_action:
                    break

    ## visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask) # visualize the graph on the image and save it
    

def run_free_explore(env,seq,fpn_feature_tensor, seq_lens, mask,name, init_points, end_points,iCurb_image_index,num):
    r'''
        The function to run free exploration and add generated data samples
        into the Dagger buffer for later training.
    '''

    agent = env.agent
    network = env.network
    args = env.args

    def visualization_image(graph_record,epoch,iCurb_image_index,name,mask,num):
        graph_record = Image.fromarray(graph_record[0,0,:,:].cpu().detach().numpy() * 255).convert('RGB')
        mask = Image.fromarray(mask * 255).convert('RGB')
        dst = Image.new('RGB', (graph_record.width * 2, graph_record.height ))
        dst.paste(graph_record,(0,0))
        dst.paste(mask,(mask.width,0))
        dst.save('./records/train/vis/free_exploration/{}_{}_{}_{}.png'.format(epoch,iCurb_image_index,name,num))

    train_len = network.train_len()
    # load data
    init_points = [[int(x[0]),int(x[1])] for x in init_points]
    end_points = [[int(x[0]),int(x[1])] for x in end_points]
    instance_num = seq.shape[0]


    # init environment
    env.init_image() # initialize the environment for a new image
    for instance_id in range(instance_num): # iterate through all curb instances in the image

        # ========================= working on a curb instance =============================
        instance_vertices = seq[instance_id] # get the vertices of the current curb instance
        agent.instance_vertices = instance_vertices[:seq_lens[instance_id]].copy() # set the vertices of the current curb instance to the agent

        if len(agent.instance_vertices):
            init_vertex = init_points[instance_id] # get the initial vertex of the current curb instance
            # init_vertex =  init_vertex + 0 * np.random.normal(0, 1, 2)
            agent.init_agent(init_vertex) # initialize the agent with the initial vertex
            agent.end_vertex = end_points[instance_id] # set the end vertex of the agent

            while 1:
                agent.agent_step_counter += 1
                # crop rectangle centering v_now
                cropped_feature_tensor = agent.crop_attention_region(fpn_feature_tensor)

                with torch.no_grad():
                    v_now = [x/1000 for x in agent.v_now]
                    v_now = torch.FloatTensor(v_now).unsqueeze(0).to(args.device)

                    v_previous = [x/1000 for x in agent.v_previous]
                    v_previous = torch.FloatTensor(v_previous).unsqueeze(0).to(args.device)

                    pre_stop_action = network.decoder_stop(cropped_feature_tensor,v_now,v_previous) # predict the stop action
                    pre_coord = network.decoder_coord(cropped_feature_tensor,v_now,v_previous) # predict the next coordinate
                pre_stop_action = pre_stop_action.squeeze(1) # remove unnecessary dimensions
                pre_coord = pre_coord.squeeze(0).squeeze(0)  # remove unnecessary dimensions
                
                env.expert_free_exploration(pre_coord,cropped_feature_tensor=cropped_feature_tensor) # run the expert policy for free exploration

                # stop action
                if (agent.agent_step_counter > args.max_length) \
                    or (((agent.v_now[0]>=999) or (agent.v_now[0]<=0)or (agent.v_now[1]>=999) or (agent.v_now[1]<=0)) and agent.agent_step_counter > 10): # if the agent has reached the maximum number of steps or the boundary of the image
                        agent.taken_stop_action = 1
                if agent.taken_stop_action:
                    break
    # visualization
    if args.visualization:
        visualization_image(env.graph_record,env.epoch_counter,iCurb_image_index,name,mask,num) # visualize the graph on the image and save it


# ------------------------------------------------------------------------------
# Exploration Modes in iCurb Training
#
# The iCurb training process includes two types of exploration strategies:
#
# 1. Restricted Exploration (`run_restricted_explore`):
#    - Follows annotated curb-line sequences from the dataset.
#    - Uses expert supervision (`expert_restricted_exploration`) to guide the agent.
#    - Only predicts the next coordinate (no stop action).IMPORTANT DIFFERENCE WITH FREE EXPLORE
#    - Stops when the curb-line is fully traversed or after 1000 steps.
#    - Produces high-quality supervised samples for training.
#
# 2. Free Exploration (`run_free_explore`):
#    - Allows the agent to explore freely without relying on curb-line annotations.
#    - Uses expert correction (`expert_free_exploration`) but the agent acts autonomously.
#    - Predicts both the next coordinate and the stop action.
#    - Stops when reaching the image boundary, exceeding max steps, or after 10+ steps.
#    - Helps the model generalize to unseen or less structured scenarios.
#
# Both functions populate the DAgger buffer with training samples and optionally
# generate visualizations of the exploration paths for debugging and analysis.
# ------------------------------------------------------------------------------


