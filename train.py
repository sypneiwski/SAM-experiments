import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MLP
from data import ModularAdditionDataset

torch.random.manual_seed(42)

def random_test(model, modulus, num_samples=20):
    """
    returns accuracy of model on random test data
    """
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, modulus, (num_samples, 2)).float()
        y = torch.sum(x, dim=1) % modulus
        y_pred = model(x)
        correct = torch.sum(torch.argmax(y_pred, dim=1) == y).item()
    return correct / num_samples

def train():
    # Hyperparameters
    modulus = 10
    batch_size = modulus**2
    epochs = 15_000
    learning_rate = 0.001

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets and dataloaders
    train_dataset = ModularAdditionDataset(modulus)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the model, loss function, and optimizer.
    model = MLP(input_dim=2, hidden_dim=32, output_dim=modulus).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop.
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_dataset)

        if epoch % 20 == 0:
            acc = random_test(model, modulus)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {acc:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # we can do early stopping for small modulus, cos the batch loss is the entire dataset
        if avg_loss < 1e-5:
            break

if __name__ == "__main__":
    train()