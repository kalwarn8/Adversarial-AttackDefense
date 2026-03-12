import torch
import torch.nn.functional as F

def fgsm_attack(model, X, y, epsilon=0.2):

    X.requires_grad = True

    outputs = model(X)

    loss = F.cross_entropy(outputs, y)

    model.zero_grad()

    loss.backward()

    grad = X.grad.data

    perturbed = X + epsilon * grad.sign()

    return perturbed.detach()