import torch
import segmentation_models_pytorch as smp

def compute_accuracy_loss(model, data_loader, criterion, device="cuda"):
    accuracy = 0
    loss = 0
    with torch.no_grad():

        for i, (image, mask) in enumerate(data_loader):

            image = image.float().to(device)
            mask = mask.float().to(device)

            mask_pred = model(image)

            loss_batch = criterion(mask_pred,mask)
            loss += loss_batch.item()

            tp, fp, fn, tn = smp.metrics.get_stats(mask_pred, mask.int(),
                                             threshold=0.5,
                                             mode='binary')
            accuracy_batch = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
            accuracy += accuracy_batch
    return accuracy / len(data_loader), loss / len(data_loader)
    


def compute_accuracy(model, data_loader, device="cuda"):
    accuracy = 0
    with torch.no_grad():

        for i, (image, mask) in enumerate(data_loader):

            image = image.float().to(device)
            mask = mask.to(device)

            mask_pred = model(image)
            tp, fp, fn, tn = smp.metrics.get_stats(mask_pred, mask,
                                             threshold=0.5,
                                             mode='binary')
            accuracy_batch = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()

            accuracy += accuracy_batch
    return accuracy / len(data_loader)
    
    
def compute_loss(model, data_loader, criterion, device="cuda"):

    loss = 0
    with torch.no_grad():
        for i, (image, mask) in enumerate(data_loader):
            image = image.float().to(device)
            mask = mask.float().to(device)

            mask_pred = model(image)
            loss_batch = criterion(mask_pred,mask)

            loss += loss_batch.item()

    return loss / len(data_loader)
