import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MLP
from data import ModularAdditionDataset
from sam import SAM

torch.random.manual_seed(42)

# Global hyperparameters
MODULUS = 13
EPOCHS = 30_000
LEARNING_RATE = 1e-3

def full_test(model, modulus):
    """
    returns accuracy of model on random test data
    """
    model.eval()
    x = torch.tensor([(a, b) for a in range(modulus) for b in range(modulus)]).float()
    y = (torch.sum(x, dim=1) % modulus).long().to(x.device)
    assert len(x) == len(y) == modulus**2
    with torch.no_grad():
        y_pred = model(x)
        correct = torch.sum(torch.argmax(y_pred, dim=1) == y).item()
    return correct / (modulus**2)

def train(train_loader, optim_class):
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model, loss function, and optimizer.
    model = MLP(input_dim=2, hidden_dim=32, output_dim=MODULUS).to(device)
    criterion = nn.CrossEntropyLoss()
    if optim_class == SAM:
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, lr=LEARNING_RATE)
    else:
        optimizer = optim_class(model.parameters(), lr=LEARNING_RATE)

    best_acc = None

    # Training loop.
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if optim_class == SAM:
                # SAM requires two backward passes
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

        if epoch % 100 == 0:
            acc = full_test(model, MODULUS)
            best_acc = max(acc, best_acc) if best_acc is not None else acc
            if acc == 1.0:
                return epoch, best_acc
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} Accuracy: {acc:.4f}")
    return None, best_acc

if __name__ == "__main__":
    noise_settings = [0.0, 0.1]
    batch_sizes = [MODULUS**2 // 2, MODULUS**2]
    optimizers = [
        # torch.optim.SGD,
        torch.optim.Adam,
        torch.optim.AdamW,
        SAM, # TODO: Explicitly explore n-SAM vs m-SAM
    ]

    res = []
    for noise_std in noise_settings:
        for batch_size in batch_sizes:
            print(f"Training with batch size {batch_size}, noise std {noise_std}")
            train_dataset = ModularAdditionDataset(MODULUS, noise_std=noise_std)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for optimizer in optimizers:
                print(f"Training with {optimizer.__name__}")
                epochs_to_solve, best_acc = train(train_loader, optim_class=optimizer)
                res.append((batch_size, optimizer, epochs_to_solve, best_acc))

    for batch_size, optimizer, epochs_to_solve, best_acc in res:
        print(f"({batch_size}) Model solved in {epochs_to_solve} epochs with {optimizer.__name__} with best accuracy {best_acc:.4f}")