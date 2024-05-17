'''This file contains code for performing multi hardware training
   The user can specify the configuration, train for 2 epochs and 
   switch the configuration again to perform further training.'''

import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.utils.data as Data
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import Fake127QPulseV1, Fake27QPulseV1, Fake5QV1, Fake20QV1, Fake7QPulseV1, GenericBackendV2
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.library import (
    IGate, CXGate, RZGate, SXGate,
    XGate, U1Gate, U2Gate, U3Gate,
    Reset, Measure, CZGate
)
from qiskit.circuit import Delay
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit.primitives import Sampler, BackendSampler
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel
import time
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from sklearn.svm import SVC
from datetime import datetime
from qiskit.converters import circuit_to_dag, dag_to_circuit
from collections import OrderedDict
from tqdm import tqdm
import json
import inspect
from utils import *
from functools import partial

# Define fixed attributes
model_dir = 'trained_models/3. multi_hardware_training'
dataset_str = input("Enter dataset string (iris/digits01/digits89):")
assert dataset_str == 'iris' or dataset_str == 'digits01' or dataset_str == 'digits89', "Input correct dataset name"
config = input("Enter configuration(qubits:20/27/127, layout:I/II/III/IV/V, eg: '27 I'):")
config_list = config.split()
assert len(config_list) == 2 and (config_list[0] == '20' or config_list[0] == '27' or config_list[0] == '127')\
    and (config_list[1] == 'I' or config_list[1] == 'II' or config_list[1] == 'III' or config_list[1] == 'IV' or config_list[1] == 'V')\
    , "Configuration data not entered properly, please try again."
num_layers = 6 if dataset_str == 'iris' else 3
r = 1 if dataset_str == 'iris' else 3 if dataset_str == 'digits01' else 1
batch_size = 16

# Load iris dataset
if dataset_str == 'iris':
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target

# Load the digits dataset
if dataset_str == 'digits01' or dataset_str == 'digits89':
    digits_data = load_digits()
    features = digits_data.data
    labels = digits_data.target

    # Specify the target labels
    target_labels = [0,1] if dataset_str == 'digits01' else [8,9]
    label_mapping = {}
    cnt = 0
    for label in target_labels:
        label_mapping[label] = cnt
        cnt+=1

    # Filter the dataset for the desired digits
    filtered_indices = np.isin(labels, target_labels)
    features_filtered = features[filtered_indices]
    labels_filtered = labels[filtered_indices]

    # Initialize empty arrays for the limited features and labels
    features_limited = np.empty((0, features_filtered.shape[1]))
    labels_limited = np.array([])

    # Limit to 50 images per specified class
    for digit in target_labels:
        digit_indices = np.where(labels_filtered == digit)[0][:]  # Get the first 100 indices for each digit
        features_limited = np.vstack((features_limited, features_filtered[digit_indices]))
        digit_labels = labels_filtered[digit_indices]
        mapped_labels = np.array([label_mapping[label] for label in digit_labels])
        labels_limited = np.concatenate((labels_limited, mapped_labels))

    features = features_limited
    labels = labels_limited


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_train), torch.from_numpy(y_test)
train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# Define the classical device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 8 QUBIT LAYOUTS
#*****************************************************************
if config_list[0] == '27':
    '''27 qubit h/w layouts'''
    if config_list[1] == 'I':
        layout_qubits = [4,7,10,12,15,18,20,23] 
        layout_coupling = [[4,7],[7,10],[10,12],[12,15],[15,18],[18,20],[20,23]]
        backend = Fake27QPulseV1()
    elif config_list[1] == 'II':
        layout_qubits = [4,7,10,12,13,15,18,20] 
        layout_coupling = [[4,7],[7,10],[10,12],[12,15],[15,18],[18,20],[12,13]]
        backend = Fake27QPulseV1()
    elif config_list[1] == 'III':
        layout_qubits = [7,10,12,15,18,20,13,14] 
        layout_coupling = [[7,10],[10,12],[12,15],[15,18],[18,20],[12,13],[13,14]]
        backend = Fake27QPulseV1()
    elif config_list[1] == 'IV':
        layout_qubits = [10,12,15,18,20,23,24,13] 
        layout_coupling = [[10,12],[12,15],[15,18],[18,20],[20,23],[23,24],[12,13]]
        backend = Fake27QPulseV1()
    elif config_list[1] == 'V':
        layout_qubits = [4,7,10,12,15,18,6,13] 
        layout_coupling = [[4,7],[7,10],[10,12],[12,15],[15,18],[6,7],[12,13]]
        backend = Fake27QPulseV1()
elif config_list[0] == '127':
    '''127 qubit h/w layouts'''
    if config_list[1] == 'I':
        layout_qubits = [14,18,19,20,21,22,23,24]
        layout_coupling = [[14,18],[18,19],[19,20],[20,21],[21,22],[22,23],[23,24]]
        backend = Fake127QPulseV1()
    elif config_list[1] == 'II':
        layout_qubits = [19,20,21,22,23,24,25,15]
        layout_coupling = [[19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[22,15]]
        backend = Fake127QPulseV1()
    elif config_list[1] == 'III':
        layout_qubits = [20,21,22,23,24,25,15,4]
        layout_coupling = [[20,21],[21,22],[22,23],[23,24],[24,25],[22,15],[15,4]]
        backend = Fake127QPulseV1()
    elif config_list[1] == 'IV':
        layout_qubits = [14,18,19,20,21,22,23,15]
        layout_coupling = [[14,18],[18,19],[19,20],[20,21],[21,22],[22,23],[22,15]]
        backend = Fake127QPulseV1()
    elif config_list[1] == 'V':
        layout_qubits = [19,20,21,22,23,24,15,33]
        layout_coupling = [[19,20],[20,21],[21,22],[22,23],[23,24],[20,33],[22,15]]
        backend = Fake127QPulseV1()
elif config_list[0] == '20':
    '''20 qubit h/w layouts'''
    if config_list[1] == 'I':
        layout_qubits = [0,1,6,5,10,11,15,16]
        layout_coupling = [[0,1],[1,6],[6,5],[5,10],[10,11],[11,16],[16,15]]
        backend = Fake20QV1()
    elif config_list[1] == 'II':
        layout_qubits = [5,6,7,8,9,14,13,3]
        layout_coupling = [[5,6],[6,7],[7,8],[8,9],[9,14],[14,13],[8,3]]
        backend = Fake20QV1()   
    elif config_list[1] == 'III':
        layout_qubits = [6,7,8,9,14,13,3,2]
        layout_coupling = [[6,7],[7,8],[8,9],[9,14],[14,13],[8,3],[3,2]]
        backend = Fake20QV1()
    elif config_list[1] == 'IV':
        layout_qubits = [0,1,2,3,8,9,14,6]
        layout_coupling = [[0,1],[1,2],[2,3],[3,8],[8,9],[9,14],[1,6]]
        backend = Fake20QV1()
    elif config_list[1] == 'V':
        layout_qubits = [5,6,1,2,3,4,0,8]
        layout_coupling = [[5,6],[6,1],[1,2],[2,3],[3,4],[1,0],[3,8]]
        backend = Fake20QV1()
#*****************************************************************

num_qubits = len(layout_qubits)

# Create the subset backend
noise_model_partial, custom_backend = create_subset_backend(layout_qubits, layout_coupling, backend)
new_lq, new_lc = qubit_numbering_mapping(layout_qubits, layout_coupling)

# Define the quantum device
dev = qml.device('qiskit.aer', wires=num_qubits, backend=custom_backend, initial_layout=new_lq)

# Define the PQC circuit
@qml.qnode(dev, interface='torch')
def pqc_iris_strong(inputs, params):
    qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits), ranges=[r]*num_layers)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev, interface='torch')
def pqc_digits_strong(inputs, params):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits), ranges=[r]*num_layers)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

# Define the model
weight_shapes = {'params': (num_layers, num_qubits, 3)}
torch.manual_seed(61)
qlayer = qml.qnn.TorchLayer(pqc_iris_strong, weight_shapes, init_method=nn.init.normal_) if dataset_str == 'iris' else \
      qml.qnn.TorchLayer(pqc_digits_strong, weight_shapes, init_method=nn.init.normal_)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
    
    def forward(self, x):
        out = self.qlayer(x)
        return out

model = Model().to(device)

'''Use this only when model is already trained for few epochs once and you are training the same model'''
# model.load_state_dict(torch.load(f'{model_dir}/strongly_entangling_layers_{dataset_str}_multi_hw_all_hw_range{r}_8q.pth'))

# Define the epochs, loss function and optimizer
epochs = 2
loss_fn = nn.CrossEntropyLoss()
soft_out = nn.Softmax(dim=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
save_steps = 1

# Define the training loop
for epoch in range(epochs):
    train_acc = test_acc = 0
    loss_list = []    
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        loss = loss_fn(soft_outputs, labels.long())
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        dist = torch.abs(labels - pred)
        train_acc += len(dist[dist==0])
    for i, (inputs, labels) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        soft_outputs = soft_out(outputs)
        pred = torch.argmax(soft_outputs, dim=1)
        dist = torch.abs(labels - pred)
        test_acc += len(dist[dist==0])
    if (epoch+1)% save_steps == 0:
        torch.save(model.state_dict(), f'{model_dir}/strongly_entangling_layers_{dataset_str}_multi_hw_all_hw_range{r}_8q.pth')
    print(f"Epoch {epoch+1}: Loss = {sum(loss_list)/len(loss_list):.4f}, Train Acc = {train_acc/len(X_train)*100:.4f}, Test Accuracy = {test_acc/len(X_test)*100:4f}")
