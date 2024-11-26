import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt

# My Custom ansatz design
def my_ansatz(num_qubits, reps):
    ansatz = QuantumCircuit(num_qubits)
    parameters = ParameterVector('θ', length=num_qubits * reps * 2)  # Custom parameters

    param_idx = 0  
    for rep in range(reps):
        for i in range(num_qubits):
            ansatz.ry(parameters[param_idx], i)  # RY rotation on each qubit
            param_idx += 1
            ansatz.rz(parameters[param_idx], i)  # RZ rotation on each qubit
            param_idx += 1
        
        # Entanglement
        for i in range(num_qubits - 1):
            ansatz.cx(i, i + 1)
        ansatz.cx(num_qubits - 1, 0)  

    return ansatz, len(parameters)  



# Fetch dataset 
ionosphere = fetch_ucirepo(id=52)

# Data (features and targets)
X = np.array(ionosphere.data.features)
y = np.array(ionosphere.data.targets)

# Convert string labels ('b', 'g') to numeric (0, 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use PCA for dimensionality reduction
pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X_scaled)

# Split into Training+Validation (80%) and Testing (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Split Training+Validation into Training (70%) and Validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

# Create feature map 
featuremap = ZZFeatureMap(feature_dimension=4, reps=2)
num_inputs = featuremap.num_parameters

# Create ansatz circuit
my_ansatz_circuit, num_weights = my_ansatz(4, 2)

# Optimizer
optimizer = COBYLA(maxiter=100)

# Using Sampler
sampler = Sampler()

# Debugging output to verify counts
print(f"Number of input parameters: {num_inputs}")
print(f"Number of weight parameters: {num_weights}")
print(f"Total parameters in circuit: {num_inputs + num_weights}")

# Initialize VQC using the feature map and ansatz
vqc = VQC(
    feature_map=featuremap,
    ansatz=my_ansatz_circuit,
    optimizer=optimizer,
    sampler=sampler
)

# Fit the model on the training data
vqc.fit(X_train, y_train)

# Predict on validation data
y_val_pred = vqc.predict(X_val)

# Calculate evaluation metrics
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

