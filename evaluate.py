import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        net = net.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        # print("image size, mean: ", image.size(), image.mean())
        mask_true = mask_true.to(device=device, dtype=torch.long)

        # predict the mask
        # print("In eval: ",net)
        # for name, param in net.named_parameters():
        #     if torch.any(param != 0):
        #         all_zero = False
        #         print(f"[OK] {name} contains non-zero values.")
        #     else:
        #         print(f"[WARNING] {name} is all zeros!")
        mask_pred = net(image)

        if net.n_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            # print("mask_pred min mean max:", mask_pred.min(), mask_pred.mean(), mask_pred.max())
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            # print(mask_true.size(), mask_true[:, 1:].mean())
            # print(mask_pred.size(), mask_pred[:, 1:].mean()) # why the pred is 0 
            # ###SOLUTION: because the model using amp to train, and the dtype is float16. Now. use float32 all the time
            # print("dice_score", dice_score)

    net.train()
    return dice_score / max(num_val_batches, 1)
