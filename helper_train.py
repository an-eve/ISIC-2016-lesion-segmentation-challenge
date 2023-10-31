import time
import torch
from tqdm.auto import tqdm
from helper_evaluation import compute_accuracy_loss


def train_model(model, num_epochs, train_loader, valid_loader, criterion,
                optimizer, device, scheduler=None, scheduler_on='valid_loss'): # valid_loss or train_loss

    train_loss_list, valid_loss_list, valid_acc_list = [], [],  []

    for epoch in tqdm(range(num_epochs)):

        batch_loss_list = []
        model.train()
        for batch_idx, (image, mask) in enumerate(tqdm(train_loader)):


            image = image.float().to(device)
            mask = mask.float().to(device)

            # ## FORWARD AND BACK PROP
            mask_pred = model(image)
            loss = criterion(mask_pred,mask)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            batch_loss_list.append(loss.item())

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_loss =  sum(batch_loss_list) / len(batch_loss_list)
            valid_acc, valid_loss = compute_accuracy_loss(model, valid_loader, criterion, device=device)
            print(f'Epoch: {epoch+1:02d}/{num_epochs:02d} '
                  f'| Train Loss: {train_loss :.4f} '
                  f'| Validation Loss: {valid_loss :.4f} '
                  f'| Validation Accuracy: {valid_acc*100 :.1f}%')
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

        if scheduler is not None:

            if scheduler_on == 'valid_loss':
                scheduler.step(valid_loss_list[-1])
            elif scheduler_on == 'train_loss':
                scheduler.step(train_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

    return train_loss_list, valid_loss_list, valid_acc_list
