import torch
import torch.optim as optim
import torch.nn.functional as F
from model import HeartDiseaseModel
from data_loader import load_fed_heart_disease
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, x, y, device):
    model.train()
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    optimizer.zero_grad()
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test_model(model, x, y, device):
    model.eval()
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(x)
        preds = outputs.argmax(dim=1)
    acc = accuracy_score(y.cpu(), preds.cpu())
    return acc

def federated_avg(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients_x, clients_y, test_x, test_y = load_fed_heart_disease()
    input_dim = clients_x[0].shape[1]

    global_model = HeartDiseaseModel(input_dim).to(device)
    epochs = 10
    local_epochs = 3
    lr = 0.01

    global_accuracies = []
    average_train_losses = []
    clients_train_losses = {i: [] for i in range(len(clients_x))}
    clients_accuracies = {i: [] for i in range(len(clients_x))}

    for epoch in range(epochs):
        print(f"\nGlobal Epoch {epoch+1}")

        local_models_state = []
        epoch_losses = []

        for client_idx, (x, y) in enumerate(zip(clients_x, clients_y)):
            local_model = HeartDiseaseModel(input_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # Train locally for some epochs and track losses
            local_losses = []
            for _ in range(local_epochs):
                loss = train_one_epoch(local_model, optimizer, x, y, device)
                local_losses.append(loss)

            # Aggregate losses for this client and store
            avg_local_loss = sum(local_losses) / len(local_losses)
            clients_train_losses[client_idx].append(avg_local_loss)
            epoch_losses.extend(local_losses)

            # Track client local accuracy
            local_acc = test_model(local_model, x, y, device)
            clients_accuracies[client_idx].append(local_acc)
            print(f" Client {client_idx} - Local loss: {avg_local_loss:.4f}, Local accuracy: {local_acc:.4f}")

            local_models_state.append(local_model.state_dict())

        # Federated averaging
        avg_state = federated_avg(local_models_state)
        global_model.load_state_dict(avg_state)

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        average_train_losses.append(avg_epoch_loss)

        # Evaluate global model on test data
        global_acc = test_model(global_model, test_x, test_y, device)
        global_accuracies.append(global_acc)

        print(f" Global Model Test Accuracy after epoch {epoch+1}: {global_acc:.4f}, Average train loss: {avg_epoch_loss:.4f}")

    # Plotting
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), average_train_losses, marker='o', label='Global Training Loss')
    plt.title('Global Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), global_accuracies, marker='o', color='green', label='Global Test Accuracy')
    plt.title('Global Test Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
