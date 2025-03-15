import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MLP
from data import ModularAdditionDataset
from sam import SAM

torch.random.manual_seed(42)

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

def train(optim_class):
    # Hyperparameters
    modulus = 13
    batch_size = modulus**2
    epochs = 16_000
    learning_rate = 0.001

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets and dataloaders
    train_dataset = ModularAdditionDataset(modulus)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the model, loss function, and optimizer.
    model = MLP(input_dim=2, hidden_dim=32, output_dim=modulus).to(device)
    criterion = nn.CrossEntropyLoss()
    if optim_class == SAM:
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, lr=learning_rate)
    else:
        optimizer = optim_class(model.parameters(), lr=learning_rate)

    best_acc = None

    # Training loop.
    for epoch in range(epochs):
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

        avg_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0:
            acc = full_test(model, modulus)
            best_acc = max(acc, best_acc) if best_acc is not None else acc
            if acc == 1.0:
                return epoch, best_acc
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {acc:.4f}")
    return None, best_acc

if __name__ == "__main__":
    optimizers = [
        optim.SGD,
        optim.Adam,
        optim.AdamW,
        SAM, # TODO: Explicitly explore n-SAM vs m-SAM
    ]
    res = []
    for optim in optimizers:
        print(f"Training with {optim.__name__}")
        epochs_to_solve, best_acc = train(optim_class=optim)
        res.append((optim, epochs_to_solve, best_acc))
    for optim, epochs_to_solve, best_acc in res:
        print(f"Model solved in {epochs_to_solve} epochs with {optim.__name__} with best accuracy {best_acc:.4f}")