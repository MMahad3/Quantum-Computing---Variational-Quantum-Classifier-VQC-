import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt

# My Custom Ansatz Design
def my_ansatz(num_qubits, layers):
    ansatz = QuantumCircuit(num_qubits)
    parameters = ParameterVector('θ', length=num_qubits * layers * 3)  

    param_idx = 0
    for layer in range(layers):
        # Apply RX, RY, RZ rotations to each qubit
        for qubit in range(num_qubits):
            ansatz.rx(parameters[param_idx], qubit)
            param_idx += 1
            ansatz.ry(parameters[param_idx], qubit)
            param_idx += 1
            ansatz.rz(parameters[param_idx], qubit)
            param_idx += 1
        
        # Full entanglement layer
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                ansatz.cx(i, j)  

    return ansatz, len(parameters)  

# Fetch dataset
ionosphere = fetch_ucirepo(id=52) 

# Data 
X = np.array(ionosphere.data.features)
y = np.array(ionosphere.data.targets)

# Convert string labels ('b', 'g') to numeric (0, 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use PCA for dimeensionality reduction
pca = PCA(n_components=4)
X_reduced = pca.fit_transform(X_scaled)

# Split into Training+Validation (80%) and Testing (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Split Training+Validation into Training (70%) and Validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

# Create the Pauli Feature Map
featuremap = PauliFeatureMap(feature_dimension=4, reps=2, paulis=["Z", "YY"], entanglement="full")

# My ansatz circuit
my_ansatz_circuit, num_weights = my_ansatz(4, 3) 

# Optimizer
optimizer = SPSA(maxiter=100)  # Switching to SPSA for better optimization

# Use Sampler
sampler = Sampler()


print(f"Number of input parameters: {featuremap.num_parameters}")
print(f"Number of weight parameters: {num_weights}")
print(f"Total parameters in circuit: {featuremap.num_parameters + num_weights}")

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


