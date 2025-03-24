import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MLP
from data import ModularAdditionDataset
from sam import SAM
from eval_sharpness import evaluate_m_sharpness, evaluate_hessian_sharpness
from tqdm import tqdm
import csv
import logging

torch.random.manual_seed(42)

# Global hyperparameters
MODULUS = 61
EPOCHS = 30_000
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_test(model, test_dataset):
    """
    returns accuracy of model on random test data
    """
    model.eval()
    
    with torch.no_grad():
        y_pred = model(test_dataset.x)
        correct = torch.sum(torch.argmax(y_pred, dim=1) == test_dataset.y).item()
    return correct / len(test_dataset)


def train(train_loader, optim_class):
    # Create the model, loss function, and optimizer.
    model = MLP(input_dim=2, hidden_dim=32, output_dim=MODULUS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if optim_class == SAM:
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            model.parameters(), base_optimizer=base_optimizer, lr=LEARNING_RATE
        )
    else:
        optimizer = optim_class(model.parameters(), lr=LEARNING_RATE)

    best_acc = None

    # Training loop.
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            if optim_class == SAM:
                # SAM requires two backward passes.
                # In theory we could do something fancy here like "use the loss on the
                # entire dataset to determine the weight perturbation, but then do
                # grad update based on the batch" or alternatives
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                criterion(model(inputs), targets).backward()
                optimizer.second_step(zero_grad=True)
                total_loss += loss.item() * inputs.size(0)
            else:
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        if epoch % 50 == 0:
            acc = full_test(model, MODULUS)
            best_acc = max(acc, best_acc) if best_acc is not None else acc
            if acc == 1.0:
                return epoch, best_acc
            if epoch % 1000 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Accuracy: {acc:.4f}"
                )
    return None, best_acc


def train_with_dual_loaders(
    train_dataset, test_dataset, perturb_batch_size, update_batch_size, optim_class
):
    # Create two DataLoaders: one for perturbation and one for the update step.
    perturb_loader = DataLoader(
        train_dataset, batch_size=perturb_batch_size, shuffle=True
    )
    update_loader = DataLoader(
        train_dataset, batch_size=update_batch_size, shuffle=True
    )

    # Initialize model, loss, and optimizer.
    model = MLP(input_dim=2, hidden_dim=32, output_dim=MODULUS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if optim_class == SAM:
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            model.parameters(), base_optimizer=base_optimizer, lr=LEARNING_RATE
        )
    else:
        optimizer = optim_class(model.parameters(), lr=LEARNING_RATE)

    best_acc = None

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0.0

        # Create iterators for both DataLoaders.
        perturb_iter = iter(perturb_loader)

        for inputs, targets in update_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # assume SAM
            try:
                inputs_perturb, targets_perturb = next(perturb_iter)
            except StopIteration:
                perturb_iter = iter(perturb_loader)
                inputs_perturb, targets_perturb = next(perturb_iter)
            inputs_perturb, targets_perturb = inputs_perturb.to(
                DEVICE
            ), targets_perturb.to(DEVICE)
            perturbed_outputs = model(inputs_perturb)

            perturb_loss = criterion(perturbed_outputs, targets_perturb)
            perturb_loss.backward()
            optimizer.first_step(zero_grad=True)

            update_loss = criterion(model(inputs), targets)
            update_loss.backward()
            optimizer.second_step(zero_grad=True)

            total_loss += update_loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_dataset)

        if epoch % 50 == 0:
            acc = full_test(model, test_dataset)
            best_acc = max(acc, best_acc) if best_acc is not None else acc

            if acc == 1.0:
                msharpness = evaluate_m_sharpness(model, test_dataset, criterion, optimizer, m=128)
                top_eigen, bulk_eigen = evaluate_hessian_sharpness(model, test_dataset, criterion)
                return epoch, best_acc, msharpness, top_eigen, bulk_eigen
        
            if epoch % 1000 == 0:
                msharpness = evaluate_m_sharpness(model, test_dataset, criterion, optimizer, m=128)
                top_eigen, bulk_eigen = evaluate_hessian_sharpness(model, test_dataset, criterion)
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {acc:.4f} | "
                    f"m-Sharpness: {msharpness:.4f} | "
                    f"Top-1 Eigenvalue: {top_eigen:.4f} | "
                    f"Top-1/Top-5 Eigenvalue: {bulk_eigen:.4f}"
                )
    return None, best_acc, msharpness, top_eigen, bulk_eigen


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING) # for hessian eigenthings

    noise_settings = [0.0, 0.5, 1.0, 2.0]
    update_batch_size = 1024
    # evenly spaced batch sizes, up to MODULUS**2
    batch_sizes = [i for i in range(1, MODULUS**2 + 1, MODULUS**2 // 4)]
    optimizers = [
        # torch.optim.SGD,
        # torch.optim.Adam,
        # torch.optim.AdamW,
        SAM,  # n-SAM vs m-SAM is implicitly controlled by the choice of batch size
    ]
    print(f"Running with noise settings {noise_settings}")
    print(f"Running with batch sizes {batch_sizes}")
    print(f"Running with optimizers {[o.__name__ for o in optimizers]}")

    res = []
    for noise_std in noise_settings:
        for batch_size in batch_sizes:
            print(f"Training with batch size {batch_size}, noise std {noise_std}")
            train_dataset = ModularAdditionDataset(
                MODULUS, device=DEVICE, noise_std=noise_std
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataset = ModularAdditionDataset(
                MODULUS, device=DEVICE, noise_std=0.
            )

            for optimizer in optimizers:
                print(f"Training with {optimizer.__name__}")
                # epochs_to_solve, best_acc = train(train_loader, optim_class=optimizer)
                epochs_to_solve, best_acc, msharpness, top_eigen, bulk_eigen = train_with_dual_loaders(
                    train_dataset,
                    test_dataset,
                    perturb_batch_size=batch_size,
                    update_batch_size=update_batch_size,
                    optim_class=optimizer,
                )
                res.append(
                    (noise_std, batch_size, optimizer, epochs_to_solve, best_acc, msharpness, top_eigen, bulk_eigen)
                )

    # save res to csv
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["noise_std", "batch_size", "optimizer", "epochs_to_solve", "best_acc", "m_sharpness", "top_eigen", "bulk_eigen"]
        )
        for row in res:
            writer.writerow(row)

    for noise_std, batch_size, optimizer, epochs_to_solve, best_acc, msharpness, top_eigen, bulk_eigen, in res:
        print(
            f"Noise Std: {noise_std} | "
            f"Batch Size: {batch_size} | "
            f"Optimizer: {optimizer.__name__} | "
            f"Epochs to Solve: {epochs_to_solve} | "
            f"Best Accuracy: {best_acc} | "
            f"m-Sharpness: {msharpness:.4f} | "
            f"Top-1 Eigenvalue: {top_eigen:.4f} | "
            f"Top-1/Top-5 Eigenvalue: {bulk_eigen:.4f}"
        )
