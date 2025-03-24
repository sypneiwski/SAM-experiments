import torch
from torch.utils.data import DataLoader

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


def evaluate_hessian_sharpness(model, test_dataset, criterion):
    """
    Calculates the largest eigenvalue of the Hessian of the criterion (worst-case curvature) and the bulk
    of the Hessian spectrum (ratio \frac{\lambda_1}{\lambda_5}) which, according to the SAM paper, is a proxy
    for sharpness. 
    """
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=True
    )
    eigenvals, _ = compute_hessian_eigenthings(model, test_loader,
                                                   criterion, 5, use_gpu=next(model.parameters()).device == torch.device("cuda"))
    return eigenvals[0], eigenvals[0] / eigenvals[4]

