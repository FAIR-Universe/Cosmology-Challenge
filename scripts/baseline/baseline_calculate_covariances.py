import torch
import h5py
import numpy as np
from wlchallenge.baseline.cnn import ResNet18

# Load model
model_path = "model.pth"
model = ResNet18()
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Read data from HDF5
with h5py.File(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/convergence_maps.h5", "r"
) as f:
    num_cosmologies = len(f.keys())
    covariance_matrices = []

    for i in range(num_cosmologies):
        print(i)
        realizations = []
        for j in range(100):  # Only the first 100 realizations
            map = f[f"cosmology_{i}"][f"realization_{j}"][()]
            map = map.reshape(
                1, 1, 256, 256
            )  # Reshape to match the model's input shape
            map = (map - map.mean()) / map.std()  # Simple normalization
            map_tensor = torch.tensor(map, dtype=torch.float32).to(device)
            with torch.no_grad():
                prediction = model(map_tensor).to("cpu").numpy()
            realizations.append(prediction)

        realizations = np.array(realizations).squeeze()
        cov_matrix = np.cov(
            realizations, rowvar=False
        )  # Calculate the covariance matrix
        covariance_matrices.append(cov_matrix)

# Save the covariance matrices to disk
covariance_matrices = np.array(covariance_matrices)
np.save(
    "/pscratch/sd/b/bthorne/fairuniverse/demo/covariance_matrices.npy",
    covariance_matrices,
)
