import argparse
import shutil
import os
import yaml

def get_parser():
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            parser.add_argument('--'+key, default=value, type=type(value))

    parser = argparse.ArgumentParser()
    
    ######### normal_dataset ##################################
    #with open('./dataset/config_dir.yml', 'r') as f:
    #    conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################

    ######## aug_dataset #######################################
    #with open('./aug_dataset/config_dir.yml', 'r') as f:
    #    conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################

    ######## dataset_PMM-NY #######################################
    #with open('./dataset_PMM-NY/config_dir.yml', 'r') as f:
    #   conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################

    ######## dataset_stoccarda #######################################
    #with open('./dataset_stoccarda/config_dir.yml', 'r') as f:
    #    conf = yaml.safe_load(f.read())    # load the config file
    #process(**conf)
    ############################################################

    ######## dataset_manhattan #######################################
    with open('./dataset_manhattan/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    ############################################################
    
    
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf) 

##### Aggiunta argomenti per le Loss #####

    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal', 'gaussian', 'focal_gaussian'],
                        help='Tipo di funzione di loss da usare: bce o focal')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Valore alpha per la Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Valore gamma per la Focal Loss')
    parser.add_argument('--gaussian_sigma', type=float, default=1.0,
                        help='Valore sigma per la Gaussian Distance Loss')
    parser.add_argument('--dataset_dir', type=str, default='./dataset', help='Percorso alla cartella radice del dataset (es. ./dataset_NY)')
    
##### Fine aggiunta argomenti per le Loss #####

################################################

    return parser

def check_and_add_dir(dir_path,clear=True):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if clear:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)



def update_dir_seg(args):
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/seg/tensorboard')
    check_and_add_dir('./records/seg/valid')
    check_and_add_dir('./records/seg/test')
    check_and_add_dir('./records/endpoint/test')
    check_and_add_dir('./records/endpoint/vertices')