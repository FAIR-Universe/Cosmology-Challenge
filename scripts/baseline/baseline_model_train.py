import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import h5py
from wlchallenge.baseline.cnn import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18()
model = model.to(device)
# Load data
# Read data from HDF5
with h5py.File(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/convergence_maps.h5", "r"
) as f:
    num_cosmologies = 100
    Nrealizations = 100
    maps = []
    labels = []
    for i in range(num_cosmologies):
        cosmology_group = f[f"cosmology_{i}"]
        omega_m = cosmology_group.attrs["omega_m"]
        sigma_8 = cosmology_group.attrs["sigma_8"]
        for j in range(Nrealizations):
            map = cosmology_group[f"realization_{j}"][()]
            maps.append(map)
            labels.append((omega_m, sigma_8))

maps = np.array(maps)
labels = np.array(labels)

# Reshape and normalize the data
maps = maps.reshape(maps.shape[0], 1, 512, 512)

# Split into training and validation sets
split = int(0.8 * len(maps))
train_maps, valid_maps = maps[:split], maps[split:]
train_labels, valid_labels = labels[:split], labels[split:]

# Convert to PyTorch tensors
train_data = TensorDataset(
    torch.tensor(train_maps, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.float32),
)
valid_data = TensorDataset(
    torch.tensor(valid_maps, dtype=torch.float32),
    torch.tensor(valid_labels, dtype=torch.float32),
)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# Define loss function and optimizer
loss_fn = (
    nn.MSELoss()
)  # Since we are regressing to cosmological parameters, we use MSE loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_maps, batch_labels in train_loader:
        batch_maps, batch_labels = batch_maps.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        predictions = model(batch_maps)
        loss = loss_fn(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(batch_maps)

    # Validation loop
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch_maps, batch_labels in valid_loader:
            batch_maps, batch_labels = batch_maps.to(device), batch_labels.to(device)
            predictions = model(batch_maps)
            loss = loss_fn(predictions, batch_labels)
            valid_loss += loss.item() * len(batch_maps)

    print(
        f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss / len(train_data):.4f}, Valid loss: {valid_loss / len(valid_data):.4f}"
    )

# Save the entire model
torch.save(model.state_dict(), "model.pth")
