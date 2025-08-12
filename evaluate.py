import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet.unet_model import UNet, UNet_7
from utils.data_loading import BasicDataset, CarvanaDataset, PerfusionDataset
# from torch.utils.data import Subset
from utils.dice_score import multiclass_dice_coeff
from utils.utils import calculate_metrics_for_2D_volume


"""evaluate with the validation set"""


# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_img = Path('./data_perfusion/images/')
dir_mask = Path('./data_perfusion/labels/')
dir_checkpoint = Path('./checkpoints/')


def eval_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        num_cls: int = 2,
        nom: str = ''
):
    #################### 1. Create dataset, Split into train / validation
    val_set = PerfusionDataset(dir_img, dir_mask, 1.0, split = 'val')
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Experiment:      {nom}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    # 4. Begin training
    
    model = model.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    model.eval()
    dice_score = 0
    num_val_batches = len(val_loader)
    count_val = 0
    list_results = np.zeros((num_cls-1, 4)) # 4 metrics

    with torch.no_grad():
        for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Run prediction
            output = model(image).float()  # <== Force float32 output
            probs = F.softmax(output, dim=1)  # optional, if using logits


            # Argmax to get predicted classes
            pred_class = probs.argmax(dim=1)

            # Convert to one-hot
            pred_onehot = F.one_hot(pred_class, model.n_classes).permute(0, 3, 1, 2).float()
            true_onehot = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()

            # Compute Dice score (ignoring background class 0)
            dice = multiclass_dice_coeff(pred_onehot[:, 1:], true_onehot[:, 1:], reduce_batch_first=False)
            a = calculate_metrics_for_2D_volume(pred_class.cpu().numpy(), mask_true.cpu().numpy(), 2)
            # print("a:", a)

            dice_score += dice
            list_results += a

            # count_val += 1

            # show the prediction
            # if epoch > 5 and count_val % 50 == 0:
            #     show_prediction(image, pred_class, mask_true)

    val_score = dice_score / max(num_val_batches, 1)
    logging.info(f'Validation Dice score: {val_score:.4f}')
    logging.info(f'Validation metrics: Dice, IoU, hd, assd')
    logging.info(f'Validation metrics: {list_results / max(num_val_batches, 1)}')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--experiment', '-exp', type=str, default='', help='The index of the experiments, like 001')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("validation.log"),       # <- saved to a file
            logging.StreamHandler()                    # <- printed to terminal
        ]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    ################ Change here to adapt to your data
    # n_classes is the number of probabilities you want to get per pixel, if only one label, then n_classes=2
    model = UNet_7(n_channels=1, n_classes=args.classes)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    else: 
        state_dict = torch.load(f'checkpoints/checkpoint_final_{args.experiment}.pth', map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from checkpoint_final_{args.experiment}.pth')


    model.to(device=device)

    eval_model(
        model=model,
        batch_size=args.batch_size,
        device=device,
        num_cls=args.classes,
        nom=args.experiment
    )
    
