import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet_7, UNet
from utils.utils import plot_img_and_mask, show_prediction
import matplotlib.pyplot as plt

"""
python predict.py -m checkpoints/checkpoint_final_50_007.pth -i data_perfusion/images/3_t1_5.png -o output_3_5.jpg
"""

def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    
    img = img.to(device=device, dtype=torch.float32)
    # print(f'img: {torch.unique(img)}')


    with torch.no_grad():
        output = net(img).float().cpu()
        
        # print(f'Output shape: {output.shape}, values: {torch.unique(output)}')
        
        probs = F.softmax(output, dim=1)  # optional, if using logits
        
        mask = probs.argmax(dim=1)
        # print(f'Mask shape: {mask.shape}, Mask values: {torch.unique(mask)}')

        # mask = F.interpolate(mask, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(img[0][0].cpu(), cmap='gray')
        # plt.title("Input")
        
        # plt.subplot(1, 3, 2)
        # plt.imshow(mask[0].cpu(), cmap='jet')
        # plt.title("Prediction")
        # plt.show()
    return mask[0].cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=args.classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image # for binary masks
            result = result.convert('L')  # Convert to grayscale
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')
