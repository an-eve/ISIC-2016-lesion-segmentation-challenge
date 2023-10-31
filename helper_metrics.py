import torch
import segmentation_models_pytorch as smp

    
def check_metrics(dataloader, model, device="cuda"):
    accuracy_list = []
    jaccard_list = []
    dice_list = []
    pix_acc_list = []
    specificity_list = []
    sensitivity_list = []

    with torch.no_grad():
        for image, mask in dataloader:
            image = image.to(device).type(torch.float32)
            mask = mask.to(device)
            pred = model(image)
            pred = (pred > 0.5).int()

            tp, fp, fn, tn = smp.metrics.get_stats(pred, mask,
                                             threshold=0.5,
                                             mode='binary')
            intersection = torch.logical_and(pred, mask).sum().detach()
            union = torch.logical_or(pred, mask).sum().detach()

            sensitivity = smp.metrics.functional.sensitivity(tp, fp, fn, tn, reduction="micro")
            specificity = smp.metrics.functional.specificity(tp, fp, fn, tn, reduction="micro")
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)


            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
            accuracy_list.append(accuracy)

            jaccard_score = intersection / union
            jaccard_list.append(jaccard_score)

            dice_score = (2.0 * intersection) / (pred.sum().detach() + mask.sum().detach())
            dice_list.append(dice_score)
            

    sensitivity = torch.mean(torch.stack(sensitivity_list)).item()
    specificity = torch.mean(torch.stack(specificity_list)).item()
    print(f"Sensitivity: {sensitivity:.6f}")
    print(f"Specificity: {specificity:.6f}")

    pix_acc = torch.mean(torch.stack(accuracy_list)).item()
    print(f"Pixel Accuracy: {pix_acc:.6f}")

    jaccard = torch.mean(torch.stack(jaccard_list)).item()
    print(f"Jaccard Score: {jaccard:.6f}")

    dice = torch.mean(torch.stack(dice_list)).item()
    print(f"Dice Score: {dice:.6f}")

    return sensitivity, specificity, pix_acc, jaccard, dice
