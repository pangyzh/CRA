import copy
import torch
import torch.nn.functional as F
import numpy as np

def flatten(grads):
    return torch.cat([g.view(-1) for g in grads])

def unflatten(flat_grad, ref_grads):
    recovered = []
    offset = 0
    for g in ref_grads:
        numel = g.numel()
        recovered.append(flat_grad[offset:offset+numel].view(g.shape))
        offset += numel
    return recovered

def get_mean_and_std(gradient_list):

    gradient_mean, gradient_std = [], []
    for idx in range(len(gradient_list[0])):
        layer_gradient_stack = torch.stack([gradient_list[i][idx] for i in range(len(gradient_list))], dim=0)
        gradient_mean.append(torch.mean(layer_gradient_stack, dim=0))
        gradient_std.append(torch.std(layer_gradient_stack, dim=0))
    
    return gradient_mean, gradient_std

def get_loss_n_accuracy(model, criterion, data_loader, args, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)