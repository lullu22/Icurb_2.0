import argparse
import shutil
import os
import yaml

def get_parser():
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            parser.add_argument('--'+key, default=value, type=type(value))

    parser = argparse.ArgumentParser()
    
    ######## normal_dataset ################################## (space_net_dataset)
    #with open('./dataset/config_dir.yml', 'r') as f:
    #    conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################

    ##### dataset_PPM-NY #######################################
    #with open('./dataset_PMM-NY/config_dir.yml', 'r') as f:
    #    conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################


    ##### dataset_manhattan #######################################
    with open('./dataset_manhattan/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    ############################################################

    
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf) 

    ##### Arguements added for Gaussian Distance Loss #####
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal', 'gaussian', 'focal_gaussian'],
                        help='Tipo di funzione di loss da usare: bce o focal')
    parser.add_argument('--gaussian_sigma', type=float, default=1.0,
                        help='Valore sigma per la Gaussian Distance Loss')
    ##### End of arguements added for Gaussian Distance Loss #####

    # ----------------- Experiment name ---------------------------
    parser.add_argument('--experiment_name', type=str, default='default_run',
                        help='name of the experiment, used for saving checkpoints and logs')
    # -----------------------------------------------------------
    

    return parser

def check_and_add_dir(dir_path,clear=True):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if clear:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)

def update_dir(args):
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/checkpoints',clear=False)
    check_and_add_dir('./records/tensorboard')
    check_and_add_dir('./records/train/vis/free_exploration')
    check_and_add_dir('./records/train/vis/restricted_exploration')
    check_and_add_dir('./records/valid/vis')
    check_and_add_dir('./records/valid/vertices_record')
    
def update_test_dir(args):
    check_and_add_dir('./records/test/skeleton')
    check_and_add_dir('./records/test/graph')
    check_and_add_dir('./records/test/final_vis')
    check_and_add_dir('./records/test/vertices_record')


def update_dir_seg(args):
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/seg/tensorboard',clear=False)
    check_and_add_dir('./records/seg/valid',clear=False)