import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import json # Assicurati che sia importato
from PIL import Image # Assicurati che sia importato
import torchvision.transforms.functional as tvf # Assicurati che sia importato

# --- Import moduli custom ---
from models.models_encoder import FPN
# Non importiamo DatasetiCurb qui

# --- Oggetto 'args' semplificato ---
class SimpleArgs:
    def __init__(self):
        # --- PERCORSI PER PREPROCESS.PY ---
        # Percorso alle IMMAGINI ORIGINALI .tiff del nuovo dataset
        # *** VERIFICA CHE QUESTO PERCORSO SIA ESATTO ***
        self.image_dir_tiff = './dataset_manhattan/cropped_tiff'
        # Percorso al file JSON di split del nuovo dataset
        self.data_split_file = './dataset_manhattan/data_split.json'
        # ------------------------------------
        self.test = False # Verrà aggiornato in base allo split

def get_file_list(mode, args):
    """ Legge i nomi base dei file da data_split.json per lo split specificato. """
    try:
        with open(args.data_split_file, 'r') as jf:
            json_data = json.load(jf)
    except FileNotFoundError:
        print(f"ERRORE: Impossibile trovare il file data_split.json in {args.data_split_file}")
        return [] # Restituisce lista vuota se il file non esiste

    # Estrai la lista corretta in base alla modalità
    if mode == 'train':
        return json_data.get('train', [])
    elif mode == 'valid':
        # Prendi solo i primi 150 come nel tuo dataset.py originale
        return json_data.get('valid', [])[:150]
    elif mode == 'test':
        return json_data.get('test', [])
    else:
        print(f"ATTENZIONE: Modalità '{mode}' non riconosciuta in data_split.json.")
        return []

def process_features(mode, model, device, args):
    """ Processa e salva le feature per uno split specifico (train/valid/test). """
    print(f'--------- Starting pre-processing for: {mode} ---------')

    # 1. Definisci la cartella di output per le feature .pt
    output_dir = f'./dataset_manhattan/precomputed_features/{mode}'
    os.makedirs(output_dir, exist_ok=True) # Crea la cartella se non esiste

    # 2. Ottieni la lista dei nomi base (senza estensione) dei file per questo split
    file_basenames = get_file_list(mode, args)
    if not file_basenames:
        print(f"Nessun file trovato per lo split '{mode}' nel file {args.data_split_file}.")
        return # Esce se non ci sono file per questo split

    print(f"Trovati {len(file_basenames)} file da processare per lo split '{mode}'.")

    # 3. Itera direttamente sui nomi base, carica i .tiff e salva i .pt
    with torch.no_grad(): # Disabilita calcolo gradienti per risparmiare memoria/tempo
        for basename in tqdm(file_basenames, desc=f'Processing {mode}'):
            # Costruisci il percorso del file TIFF di input
            tiff_filename = f"{basename}.tiff"
            tiff_path = os.path.join(args.image_dir_tiff, tiff_filename)

            # Costruisci il percorso del file PT di output
            pt_filename = f"{basename}.pt" # Cambia estensione
            save_path = os.path.join(output_dir, pt_filename)

            # Controllo opzionale: Salta se il file .pt esiste già
            # if os.path.exists(save_path):
            #     continue

            try:
                # --- Carica l'immagine TIFF originale ---
                img = Image.open(tiff_path)
                tiff_tensor = tvf.to_tensor(img) # Converte in tensore [C, H, W]
                # ---------------------------------------

                # Aggiungi la dimensione batch e sposta sulla GPU corretta
                tiff_tensor = tiff_tensor.unsqueeze(0).to(device) # -> [1, C, H, W]

                # --- Calcolo pesante (Encoder FPN + Interpolazione) ---
                fpn_feature_map = model(tiff_tensor)
                fpn_feature_map = F.interpolate(fpn_feature_map,
                                                scale_factor=4,
                                                mode='bilinear',
                                                align_corners=True)
                # -----------------------------------------------------

                # Salva il tensore delle feature su disco
                # .squeeze(0) rimuove la dimensione batch -> [C, H, W]
                # .cpu() sposta sulla CPU prima di salvare
                torch.save(fpn_feature_map.squeeze(0).cpu(), save_path)

            except FileNotFoundError:
                print(f"\nATTENZIONE: File TIFF non trovato: {tiff_path}. Salto questo file.")
            except Exception as e: # Cattura altri errori (es. file corrotto)
                print(f"\nErrore durante il processamento di {tiff_filename}: {e}")

    print(f'--------- Pre-processing for {mode} complete! ---------')
    print(f'Features saved to: {output_dir}\n')


# Questo blocco viene eseguito solo quando lanci 'python preprocess.py'
if __name__ == "__main__":

    # Imposta il dispositivo (GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 1. Inizializza gli argomenti semplificati
    args = SimpleArgs()

    # 2. Inizializza il modello FPN
    desired_backbone = 'efficientnet_b4'
    N_CHANNELS = 4 
    print(f"Initialization of the FPN model whit: {desired_backbone}")
    model = FPN(
        backbone_name=desired_backbone,
        n_channels=N_CHANNELS
    )

    # Carica i pesi pre-addestrati specifici per PMM-NY
    try:
        checkpoint_path = './checkpoints/seg_pretrain_manhattan_efficentnet_1.6.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Caricato checkpoint encoder da: {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERRORE CRITICO: Checkpoint encoder non trovato in {checkpoint_path}.")
        print("Assicurati che il file esista prima di eseguire preprocess.py.")
        exit() # Interrompe l'esecuzione se il checkpoint manca

    # Sposta il modello sul dispositivo corretto
    model.to(device)
    # Metti il modello in modalità valutazione (importante!)
    model.eval()

    # 3. Esegui il pre-processing per tutti e tre gli split
    process_features('train', model, device, args)
    process_features('valid', model, device, args)
    process_features('test', model, device, args)

    print("All pre-processing is complete.")