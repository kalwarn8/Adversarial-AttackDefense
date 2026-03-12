import torch

def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, iters=10):

    perturbed = X.clone().detach()

    for i in range(iters):

        perturbed.requires_grad = True

        outputs = model(perturbed)

        loss = torch.nn.functional.cross_entropy(outputs, y)

        model.zero_grad()

        loss.backward()

        grad = perturbed.grad.data

        perturbed = perturbed + alpha * grad.sign()

        eta = torch.clamp(perturbed - X, min=-epsilon, max=epsilon)

        perturbed = torch.clamp(X + eta, min=-3, max=3).detach()

    return perturbed