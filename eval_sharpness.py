import torch
from torch.utils.data import DataLoader
import math

from hessian_eigenthings import compute_hessian_eigenthings


def evaluate_m_sharpness(model, test_dataset, criterion, optimizer, m=128):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=m, shuffle=False)
    
    total_sharpness = 0.0
    num_batches = 0

    for inputs, targets in test_loader:

        # Compute original loss
        outputs = model(inputs)
        original_loss = criterion(outputs, targets)

        # Perturb the weights (gradient ascent step)
        original_loss.backward()
        optimizer.first_step(zero_grad=True)

        # Compute perturbed loss
        perturbed_outputs = model(inputs)
        perturbed_loss = criterion(perturbed_outputs, targets)

        # Compute m-sharpness
        batch_sharpness = (perturbed_loss - original_loss).item()
        total_sharpness += batch_sharpness
        num_batches += 1

        # Revert the perturbation without a second backward pass
        optimizer.second_step(zero_grad=True)

    avg_sharpness = total_sharpness / num_batches
    return avg_sharpness


def evaluate_hessian_sharpness(model, test_dataset, criterion, use_gpu=False):
    """
    Calculates the largest eigenvalue of the Hessian of the criterion (worst-case curvature) and the bulk
    of the Hessian spectrum (ratio \frac{\lambda_1}{\lambda_5}) which, according to the SAM paper, is a proxy
    for sharpness. 
    """
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=True
    )
    eigenvals, _ = compute_hessian_eigenthings(model, test_loader, criterion, 1, use_gpu=use_gpu)
    return eigenvals[0]

def evaluate_fisher_rao_norm(model, train_dataset, criterion, device='cuda'):
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )
    model = model.to(device)
    
    with torch.no_grad():
        param_vector = torch.cat([p.flatten() for p in model.parameters()])
    
    sum_of_squares = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total_samples += 1

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        model.zero_grad()
        grads = torch.autograd.grad(loss, model.parameters(),
                                    retain_graph=False,
                                    create_graph=False)
        grad_vector = torch.cat([g.flatten() for g in grads if g is not None])

        dot_val = torch.dot(grad_vector, param_vector)
        sum_of_squares += dot_val.item()**2
    
    fisher_rao_value = math.sqrt(sum_of_squares / total_samples)

    return fisher_rao_value
