import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics.segmentation import dice


@torch.inference_mode()
def evaluate(net, dataloader, device): # only for evaluate in the training process
    # iterate over the validation set
    net.eval()
    dice_score = 0
    num_val_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Run prediction
            output = net(image).float()  # <== Force float32 output
            probs = F.softmax(output, dim=1)  # optional, if using logits

            # Check output stats
            # print("Output stats:", output.min(), output.max(), output.mean())
            # print("Probs stats:", probs.min(), probs.max(), probs.mean())

            # Argmax to get predicted classes
            pred_class = probs.argmax(dim=1)

            # Convert to one-hot
            pred_onehot = F.one_hot(pred_class, net.n_classes).permute(0, 3, 1, 2).float()
            true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

            # Sanity check
            # print("pred_onehot mean (ignore bg):", pred_onehot[:, 1:].mean())
            # print("true_onehot mean (ignore bg):", true_onehot[:, 1:].mean())

            # Compute Dice score (ignoring background class 0)
            dice = multiclass_dice_coeff(pred_onehot[:, 1:], true_onehot[:, 1:], reduce_batch_first=False)

            # dice_metric = DiceScore(num_classes=net.n_classes, average='macro', ignore_index=0).to(device)
            # # during validation
            # dice_1 = dice_metric(pred_class, mask_true)
            # print("Dice compare: ", dice, dice_1)

            dice_score += dice

    net.train()
    return dice_score / max(num_val_batches, 1)

