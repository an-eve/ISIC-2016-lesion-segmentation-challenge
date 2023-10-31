import matplotlib.pyplot as plt
import numpy as np
import torch
import random


def plot_loss_accuracy(train_loss_list, val_loss_list, val_acc_list):

    num_epochs = len(train_loss_list)
    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(np.arange(1, num_epochs+1), train_loss_list, color="green", label='Train Loss')
    ax1.plot(np.arange(1, num_epochs+1), val_loss_list, color="red", label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.legend()

    ax2.plot(np.arange(1, num_epochs+1), val_acc_list, color="blue", label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def plot_img_mask_pred(dataset, index=None, plot_pred=False, model=None, device = "cuda"):
    if not index:
        index = random.randint(0, len(dataset) - 1)

    image = dataset[index][0].permute(1,2,0)
    mask = dataset[index][1].permute(1,2,0)

    if plot_pred:
        img_to_pred = dataset[index][0].unsqueeze(0).type(torch.float32).to(device)
        pred = model(img_to_pred)
        pred = pred.squeeze(0).cpu().detach().permute(1,2,0)
        pred[pred < 0]=0
        pred[pred > 0]=1


        # Plot the image
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")

        # Plot the predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Prediction")

        # Show the plots
        plt.tight_layout()
        plt.show()

    else:
        # Plot the image
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")

        # Plot the mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")

        # Show the plots
        plt.tight_layout()
        plt.show()

