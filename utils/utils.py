import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

# import matplotlib.pyplot as plt

def show_prediction(inputs, preds, gts, idx=0):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(inputs[idx][0].cpu(), cmap='gray')
    plt.title("Input")
    
    plt.subplot(1, 3, 2)
    plt.imshow(preds[idx].cpu(), cmap='jet')
    plt.title("Prediction")

    plt.subplot(1, 3, 3)
    plt.imshow(gts[idx].cpu(), cmap='jet')
    plt.title("Ground Truth")

    plt.show()
