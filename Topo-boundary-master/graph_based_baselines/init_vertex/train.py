import argparse
import json
import os
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
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from skimage.morphology import skeletonize
from models.models_encoder import *
from arguments import *
import datetime
from scipy.ndimage import distance_transform_edt
import time



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


############################## Focal Loss Implementation ##################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'): 
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_factor = alpha_t * (1 - pt) ** self.gamma
        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
#############################################################################################

########################### Focal Gaussian Loss Implementation ##############################

class FocalGaussianLoss(nn.Module):
    def __init__(self, sigma=2.0, alpha=0.85, gamma=2.0, reduction="mean",scale = 2.0):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.scale = scale  # scaling factor to balance the loss magnitude

    def forward(self, inputs, targets, already_gaussian=False):
        """
        inputs: [B,1,H,W] logit
        targets: [B,1,H,W] binario
        """
        B, _, H, W = targets.shape
        device = targets.device

        # --- Genera Gaussian heatmap ---
        if already_gaussian:
            heatmaps = targets
        else:
            targets_np = targets.detach().cpu().numpy()
            heatmaps = []
            for b in range(B):
                gt = targets_np[b,0].astype(np.uint8)
                if gt.sum() > 0:
                    dist_map = distance_transform_edt(1 - gt)
                    heatmap = np.exp(-(dist_map**2)/(2*self.sigma**2))
                else:
                    heatmap = np.zeros((H,W), dtype=np.float32)
                heatmaps.append(torch.from_numpy(heatmap).to(device))
            heatmaps = torch.stack(heatmaps).unsqueeze(1).float()

        # --- predizione sigmoid ---
        pred = torch.sigmoid(inputs)

        # --- MSE ---
        mse = (pred - heatmaps) ** 2

        # --- Focal weighting ---
        pt = torch.where(targets == 1, pred, 1 - pred)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_factor = alpha_t * (1 - pt) ** self.gamma

        # Normalizzazione per evitare NaN o scala minuscola
        focal_factor = focal_factor / (focal_factor.mean() + 1e-2)

        # --- Loss combinata ---
        loss = focal_factor * mse

        if self.reduction == "mean":
            return self.scale * loss.mean()  # scaling finale
        elif self.reduction == "sum":
            return self.scale * loss.sum()
        else:
            return loss


#############################################################################################


############################ Dataset ###################################
class dataset(Dataset):
    def __init__(self,args,valid=False):

        ######### normal_dataset #######################
        #with open('./dataset/data_split.json','r') as jf:
        ################################################

        ######## aug_dataset ###########################
        #with open('./aug_dataset/data_split.json','r') as jf:
        ################################################

        ######## space_net_dataset ###########################
        #with open('./space_net_dataset/data_split.json','r') as jf:
        ################################################

        ######## dataset_PMM-NY ###########################
        #with open('./dataset_PMM-NY/data_split.json','r') as jf:
        ###################################################

        ######## dataset_manhattan ###########################
        with open('./dataset_manhattan/data_split.json','r') as jf:
        ###################################################

            json_list = json.load(jf)
        self.file_list = json_list['pretrain'] + json_list['train']
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        self.endpoint_list = [os.path.join(args.endpoint_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the training data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        endpoint = tvf.to_tensor(Image.open(self.endpoint_list[idx]))
        return tiff,mask,endpoint

class valid_dataset(Dataset):
    def __init__(self,args):

        ######### normal_dataset #######################
        #with open('./dataset/data_split.json','r') as jf:
        ################################################

        ######## aug_dataset ###########################
        #with open('./aug_dataset/data_split.json','r') as jf:
        ################################################

        ######## space_net_dataset ###########################
        #with open('./space_net_dataset/data_split.json','r') as jf:
        ################################################

        ######## dataset_PMM-NY ###########################
        #with open('./dataset_PMM-NY/data_split.json','r') as jf:
        ###################################################

        ######## dataset_manhattan ###########################
        with open('./dataset_manhattan/data_split.json','r') as jf:
        ###################################################

            json_list = json.load(jf) 
        self.file_list = json_list['valid'][:2000]
        self.tiff_list = [os.path.join(args.image_dir,'{}.tiff'.format(x)) for x in self.file_list]
        self.mask_list = [os.path.join(args.mask_dir,'{}.png'.format(x)) for x in self.file_list]
        self.endpoint_list = [os.path.join(args.endpoint_dir,'{}.png'.format(x)) for x in self.file_list]
        print('Finish loading the valid data set lists {}!'.format(len(self.file_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self,idx):
        tiff = tvf.to_tensor(Image.open(self.tiff_list[idx]))
        mask = tvf.to_tensor(Image.open(self.mask_list[idx]))
        endpoint = tvf.to_tensor(Image.open(self.endpoint_list[idx]))
        name = self.file_list[idx]
        return tiff,mask,endpoint,name 

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

######################### Funzione per salvare tensori come immagini ########################
def save_tensor_image(tensor, filename):
    img = tensor.detach().cpu().numpy()
    img = np.squeeze(img)
    if img.ndim == 1:
        img = img.reshape(-1, 1)
    if img.ndim == 2 and img.shape[0] == 1:
        img = img.T
    elif img.ndim == 3 and img.shape[0] <= 3:
        img = np.transpose(img, (1, 2, 0))
    img = (img*255).clip(0,255).astype('uint8')
    try:
        if img.ndim == 2:
            Image.fromarray(img).save(filename)
        elif img.ndim == 3:
            Image.fromarray(img).convert('RGB').save(filename)
        else:
            print(f"Formato immagine non supportato: {img.shape}")
    except Exception as e:
        print(f"Errore nel salvataggio immagine {filename}: {e}")

################################# Train ###################################
def train(args, epoch, net, dataloader, train_len, optimizer, criterion, writer, valid_dataloader, valid_len, save_dir, checkpoint_path, valid_save_dir, initial_best_f1=0.0):
   
    net.train()
    counter = 0
    best_f1 = initial_best_f1


    epoch_start_time = time.time()

    for idx, data in enumerate(dataloader):
        img, mask, endpoint = data
        img = img.to(args.device)
        mask = mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        endpoint = endpoint[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        pred_binary_mask, pred_endpoint_map, _ = net(img)

        if args.loss_type == 'bce':
            loss_main = criterion['bce'](pred_binary_mask, mask) + criterion['bce'](pred_endpoint_map, endpoint)
        elif args.loss_type == 'focal':
            loss_main = criterion['focal'](pred_binary_mask, mask) + criterion['focal'](pred_endpoint_map, endpoint)
        elif args.loss_type == 'gaussian':
            loss_main = criterion['gaussian'](pred_binary_mask, mask, already_gaussian=False) + criterion['gaussian'](pred_endpoint_map, endpoint, already_gaussian=True)
        elif args.loss_type == 'focal_gaussian':
            loss_main = criterion['focal_gaussian'](pred_binary_mask, mask, already_gaussian=False) + criterion['focal_gaussian'](pred_endpoint_map, endpoint, already_gaussian=True)
        
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

        print(f'Epoch: {epoch}/{args.epochs} || batch: {idx}/{train_len} || loss: {round(loss_main.item(),5)}') 
        writer.add_scalar(f'train/{args.loss_type}_loss', loss_main.item(), counter + train_len*epoch)
        counter += 1

        if idx % 500 == 0:
            save_tensor_image(mask[0], os.path.join(save_dir, f'epoch{epoch}_batch{idx}_mask.png'))
            save_tensor_image(endpoint[0], os.path.join(save_dir, f'epoch{epoch}_batch{idx}_endpoint.png'))
            save_tensor_image(torch.sigmoid(pred_binary_mask[0]), os.path.join(save_dir, f'epoch{epoch}_batch{idx}_pred_mask.png'))
            save_tensor_image(torch.sigmoid(pred_endpoint_map[0]), os.path.join(save_dir, f'epoch{epoch}_batch{idx}_pred_endpoint.png'))

        if idx % (train_len-1) == 0 and idx:
            f1 = val(args, epoch, net, valid_dataloader, counter + train_len*epoch, valid_len, writer, valid_save_dir, criterion)
            if f1 > best_f1:
                best_f1 = f1
                print(f"🏆 Nuovo F1 score migliore: {f1:.4f}. Salvataggio checkpoint all'epoca {epoch}...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                }, checkpoint_path)


    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch}/{args.epochs} finished in {epoch_duration/60:.2f} minutes.")

    remain_time = (args.epochs - epoch + 1) * epoch_duration
    print(f"Estimated remaining hours: {remain_time/3600:.2f} h and minutes: {(remain_time%3600)/60:.2f} min")


################################# Validation ###################################
def val(args, epoch, net, dataloader, ii, val_len, writer, valid_save_dir, criterion):
    from skimage.morphology import skeletonize
    from scipy.spatial import cKDTree
    import numpy as np
    from PIL import Image
    import torch



    def eval_metric(seg_result, mask):
        def tuple2list(t):
            return [[t[0][x], t[1][x]] for x in range(len(t[0]))]

        skel = skeletonize(seg_result, method='lee') 
        gt_points = tuple2list(np.where(mask != 0))
        graph_points = tuple2list(np.where(skel != 0))
        gt_points = np.array(gt_points)
        graph_points = np.array(graph_points)

        if gt_points.ndim != 2 or gt_points.shape[0] == 0:
            # se il ground truth è vuoto, ritorna None
            return None, None, None
        
        if graph_points.ndim != 2 or graph_points.shape[0] == 0:
            return 0, 0, 0

        gt_tree = cKDTree(gt_points)
        graph_acc = graph_recall = 0
        for thre in [5]:
            graph_tree = cKDTree(graph_points)
            graph_dds, _ = graph_tree.query(gt_points, k=1)
            gt_acc_dds, _ = gt_tree.query(graph_points, k=1)
            graph_recall = len([x for x in graph_dds if x < thre]) / len(graph_dds)
            graph_acc = len([x for x in gt_acc_dds if x < thre]) / len(gt_acc_dds)
        f1 = 0
        if graph_acc * graph_recall:
            f1 = 2 * graph_recall * graph_acc / (graph_acc + graph_recall)
        return graph_acc, graph_recall, f1

    net.eval()
    f1_ave = 0
    valid_count = 0  # conta solo immagini con GT valido

    for idx, data in enumerate(dataloader):
        img, mask, endpoint, name = data
        img = img.to(args.device)
        mask_tensor = mask[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        endpoint_tensor = endpoint[:,0:1,:,:].type(torch.FloatTensor).to(args.device)
        mask_np = mask_tensor[0,0,:,:].cpu().numpy()

        with torch.no_grad():
            pred_mask, pred_endpoint, _ = net(img)
            pred_mask_sig = torch.sigmoid(pred_mask[0,0,:,:]).cpu().numpy()
            pred_endpoint_sig = torch.sigmoid(pred_endpoint[0,0,:,:]).cpu().numpy()

            # Normalizza asse x da 0 a 100
            x_norm = idx / (val_len - 1) * 100  # in questo caso i valori vanno da 0 a 100

            # 🔹 Loss combinata mask + endpoint
            if args.loss_type == 'bce':
                loss_val = criterion['bce'](pred_mask, mask_tensor) + criterion['bce'](pred_endpoint, endpoint_tensor)
                writer.add_scalar(f"val/bce_loss/epoch_{epoch}", loss_val.item(), x_norm)
            elif args.loss_type == 'focal':
                loss_val = criterion['focal'](pred_mask, mask_tensor) + criterion['focal'](pred_endpoint, endpoint_tensor)
                writer.add_scalar(f"val/focal_loss/epoch_{epoch}", loss_val.item(), x_norm)
            elif args.loss_type == 'gaussian':
                loss_val = criterion['gaussian'](pred_mask, mask_tensor, already_gaussian=False) + criterion['gaussian'](pred_endpoint, endpoint_tensor, already_gaussian=True)
                writer.add_scalar(f"val/gaussian_loss/epoch_{epoch}", loss_val.item(), x_norm)
            elif args.loss_type == 'focal_gaussian':
                loss_val = criterion['focal_gaussian'](pred_mask, mask_tensor, already_gaussian=False) + criterion['focal_gaussian'](pred_endpoint, endpoint_tensor, already_gaussian=True)
                writer.add_scalar(f"val/focal_gaussian_loss/epoch_{epoch}", loss_val.item(), x_norm)

            # 🔹 Salvataggio predizioni
            Image.fromarray((pred_mask_sig / np.max(pred_mask_sig) * 255).astype('uint8')).convert('RGB').save(
                os.path.join(valid_save_dir, f'{name[0]}_mask.png')
            )
            Image.fromarray((pred_endpoint_sig / np.max(pred_endpoint_sig) * 255).astype('uint8')).convert('RGB').save(
                os.path.join(valid_save_dir, f'{name[0]}_endpoint.png')
            )
            Image.fromarray((skeletonize((pred_mask_sig > 0.2).astype(np.uint8), method='lee') * 255).astype('uint8')).convert('RGB').save(
                os.path.join(valid_save_dir, f'{name[0]}_skeleton.png')
            )
            
            ######
            import matplotlib.pyplot as plt

            # Salvataggio figura su TensorBoard
            if idx in (100, 200, 300, 400) :  # salva solo per alcuni indici
                
                
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))
                axes[0].imshow(pred_mask_sig, cmap='gray')
                axes[0].set_title('Pred Mask')
                axes[0].axis('off')

                axes[1].imshow(pred_endpoint_sig, cmap='gray')
                axes[1].set_title('Pred Endpoint')
                axes[1].axis('off')

                axes[2].imshow(skeletonize((pred_mask_sig > 0.2).astype(np.uint8), method='lee'), cmap='gray')
                axes[2].set_title('Skeleton')
                axes[2].axis('off')

                axes[3].imshow(mask_np, cmap='gray')
                axes[3].set_title('GT Mask')
                axes[3].axis('off') 

                writer.add_figure(f'val/predictions/{name[0]}_epoch{epoch}', fig, global_step=epoch)
                plt.close(fig)

            ######     


            # Metriche
            acc, rec, f1 = eval_metric((pred_mask_sig > 0.2).astype(np.uint8), (mask_np > 0.2).astype(np.uint8))
            if f1 is not None:
                f1_ave = (f1_ave * valid_count + f1) / (valid_count + 1) 
                valid_count += 1
                print(f'Validation:{epoch}/{args.epochs} || Image:{idx}/{val_len} || F1:{round(f1,3)} || Accuracy:{round(acc,3)} || Recall:{round(rec,3)}') #### da valutare se tenere
            else:
                print(f'Validation:{epoch}/{args.epochs} || Image:{idx}/{val_len} || F1:None (GT vuoto)') 
                
            ii += 1

    print(f'Validation Summary:{epoch}/{args.epochs} || Average F1:{round(f1_ave,3)}')
    writer.add_scalar(f'val/{args.loss_type}_f1', f1_ave, ii)

    return f1_ave


################################# Main ###################################
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    ### change when we want try new experiment #######
    new_name = "manhattan_efficentnet_b4_1.6"
    ##################################################

    # Selezione loss
    if args.loss_type == 'focal':
        criterion = {'focal': FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)}
    elif args.loss_type == 'gaussian':
        criterion = {'gaussian': GaussianDistanceLoss(sigma=args.gaussian_sigma)}
    elif args.loss_type == 'focal_gaussian':
        criterion = {'focal_gaussian': FocalGaussianLoss(sigma=args.gaussian_sigma, alpha=args.focal_alpha, gamma=args.focal_gamma)}
    else:
        criterion = {'bce': nn.BCEWithLogitsLoss()}
    

    # Writer con run separato e timestamp
    log_dir_base = "./records/seg"
    os.makedirs(log_dir_base, exist_ok=True)
    run_name = f"{new_name}_{args.loss_type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(os.path.join(log_dir_base, run_name))

    save_dir = f'./records/train_images_{args.loss_type}_{new_name}'
    checkpoint_path = f'./checkpoints/seg_pretrain_{args.loss_type}_{new_name}.pth'
    valid_save_dir = f'./records/seg/valid_{args.loss_type}_{new_name}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(valid_save_dir, exist_ok=True)

    update_dir_seg(args)

    device = args.device

    train_dataset = dataset(args)
    valid_dataset_obj = valid_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset_obj, batch_size=1, shuffle=False)
    train_len = len(train_loader)
    valid_len = len(valid_loader)

    # Modello e ottimizzatore
    #net = FPN()
    #net = FPN(backbone_name='resnet101')
    net = FPN(backbone_name='efficientnet_b4')

    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)





    ##### checkpoint #################################


    start_epoch = 0
    best_f1 = 0.0

    checkpoint_path = f'./checkpoints/seg_pretrain_{args.loss_type}_{new_name}.pth'
   

    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint trovato! Caricamento da: {checkpoint_path}")
        # Carichiamo il checkpoint sulla CPU per evitare problemi di memoria
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Controlliamo se è il nuovo formato (dizionario) o il vecchio
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Nuovo formato
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint.get('best_f1', 0.0) # .get() per retrocompatibilità
            print(f"Checkpoint completo caricato. Si riparte dall'epoca {start_epoch}.")
        else:
            # Vecchio formato (solo pesi)
            net.load_state_dict(checkpoint)
            start_epoch = 0 # Ripartiamo a contare le epoche, ma con i pesi già addestrati
            print("Checkpoint legacy (solo pesi) caricato. L'addestramento riprende con pesi pre-addestrati.")
    else:
        print("Nessun checkpoint trovato. Inizio dell'addestramento da zero.")


    #################################################

    # Training loop
    for epoch in range(args.epochs):
        train(args, epoch, net, train_loader, train_len, optimizer, criterion, writer,
              valid_loader, valid_len, save_dir, checkpoint_path, valid_save_dir)
