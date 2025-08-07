import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import json
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, PerfusionDataset
from utils.dice_score import dice_loss
# from torch.utils.data import Subset
from utils.dice_score import multiclass_dice_coeff
from utils.utils import plot_img_and_mask, show_prediction

from torch.utils.tensorboard import SummaryWriter



# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_img = Path('./data_perfusion/images/')
dir_mask = Path('./data_perfusion/labels/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        writer,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    #################### 1. Create dataset, Split into train / validation
    train_set = PerfusionDataset(dir_img, dir_mask, img_scale, split = 'train') # dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    val_set = PerfusionDataset(dir_img, dir_mask, img_scale, split = 'val')
    
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
                            #   lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    global_step = 0
    val_max = 0

    # 4. Begin training
    for epoch in range(1, epochs + 1):
        model = model.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # print("#### in training: images: size, mean",images.size(),images.mean())
                masks_pred = model(images)
                # print("#### in training: masks_pred: size, mean",masks_pred.size(),masks_pred.cpu().unique())
                # print("true-masks: ",true_masks.size(),true_masks.cpu().unique())
                # exit()

                loss = criterion(masks_pred, true_masks)
                # print("loss before",loss)
                loss += dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                # print("loss after",loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log to tensorboard
                writer.add_scalar("Loss/train", loss.item() , global_step)
                writer.add_scalar("LearningRate/train", optimizer.param_groups[0]['lr'], global_step)

                ################## Evaluation round
                division_step = (n_train // (4 * batch_size)) # every 25 % of an epoch
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score = evaluate(model, val_loader, device)
                        # scheduler.step(val_score)
                        model.eval()
                        dice_score = 0
                        num_val_batches = len(val_loader)
                        count_val = 0

                        with torch.no_grad():

                            for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                                image, mask_true = batch['image'], batch['mask']
                                image = image.to(device=device, dtype=torch.float32)
                                mask_true = mask_true.to(device=device, dtype=torch.long)

                                # Run prediction
                                output = model(image).float()  # <== Force float32 output
                                probs = F.softmax(output, dim=1)  # optional, if using logits

                                # Check output stats
                                # print("Output stats:", output.min(), output.max(), output.mean())
                                # print("Probs stats:", probs.min(), probs.max(), probs.mean())

                                # Argmax to get predicted classes
                                pred_class = probs.argmax(dim=1)

                                # Convert to one-hot
                                pred_onehot = F.one_hot(pred_class, model.n_classes).permute(0, 3, 1, 2).float()
                                true_onehot = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()

                                # Sanity check
                                # print("pred_onehot mean (ignore bg):", pred_onehot[:, 1:].mean())
                                # print("true_onehot mean (ignore bg):", true_onehot[:, 1:].mean())

                                # Compute Dice score (ignoring background class 0)
                                dice = multiclass_dice_coeff(pred_onehot[:, 1:], true_onehot[:, 1:], reduce_batch_first=False)

                                dice_score += dice
                                count_val += 1

                                # show the prediction
                                if epoch > 5 and count_val % 50 == 0:
                                    show_prediction(image, pred_class, mask_true)

                        model.train()
                        val_score = dice_score / max(num_val_batches, 1)
                        logging.info(f'Validation Dice score: {val_score:.4f}')

                        # Save the best model based on validation score
                        if val_score>val_max:
                            # save the checkpoints
                            val_max = val_score
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            state_dict['mask_values'] = train_set.mask_values #dataset.mask_values
                            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_max_Dice.pth'))
                            logging.info(f'Checkpoint {epoch} saved for it max dice in val set!')
                        
                        writer.add_scalar("Dice/val", val_score, global_step)

                        logging.info(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Dice={val_score:.4f}")

                        

        if epoch == epochs: 
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values #dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_final_{}.pth'.format(epoch)))
            logging.info(f'Final checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--experiment', '-exp', type=str, default='', help='The index of the experiments, like 001')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training.log"),       # <- saved to a file
            logging.StreamHandler()                    # <- printed to terminal
        ]
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    writer = SummaryWriter(log_dir=f'runs/experiment_{args.experiment}')

    ################ Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_channels=1 for grayscale images
    # n_classes is the number of probabilities you want to get per pixel, if only one label, then n_classes=2
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
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

    model.to(device=device)
    try:
        train_model(
            model=model,
            writer = writer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            # val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            writer = writer,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            # val_percent=args.val / 100,
            amp=args.amp
        )
