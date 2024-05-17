'''This file contains code to train the motivational example shown in
   Fig. 1 of the paper. The training is done on single configuration
   for 10 epochs.'''

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
model_dir = 'trained_models/1. motivation'
example_number = input("Enter example number for motivation (1/2/3)")
assert example_number == '1' or example_number == '2' or example_number == '3',"Only 3 motivational examples available, select 1/2/3."
num_layers = 3
batch_size = 16

# Load the digits dataset
digits_data = load_digits()
features = digits_data.data
labels = digits_data.target

# Specify the target labels
target_labels = [8,9]
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
'''Motivation layouts from 127 qubit hardware (you can pick any one)'''

if example_number == '1':
    layout_qubits = [9,10,11,12,17,30,29,28]
    layout_coupling = [[9,10],[10,11],[11,12],[12,17],[17,30],[30,29],[29,28]]
    backend = Fake127QPulseV1()
elif example_number == '2':
    layout_qubits = [79,80,81,82,83,84,85,86]
    layout_coupling = [[79,80],[80,81],[81,82],[82,83],[83,84],[84,85],[85,86]]
    backend = Fake127QPulseV1()
elif example_number == '3':
    layout_qubits = [97,98,99,100,110,118,119,120]
    layout_coupling = [[97,98],[98,99],[99,100],[100,110],[110,118],[118,119],[119,120]]
    backend = Fake127QPulseV1()
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
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits), ranges=[1]*num_layers)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

@qml.qnode(dev, interface='torch')
def pqc_digits_strong(inputs, params):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(6), normalize=True)
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits), ranges=[1]*num_layers)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

# Define the model
weight_shapes = {'params': (num_layers, num_qubits, 3)}
torch.manual_seed(61)
qlayer = qml.qnn.TorchLayer(pqc_digits_strong, weight_shapes, init_method=nn.init.normal_)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
    
    def forward(self, x):
        out = self.qlayer(x)
        return out

model = Model().to(device)

# Define the epochs, loss function and optimizer
epochs = 10
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
        torch.save(model.state_dict(), f'{model_dir}/digits89_range1_127q_motivation_{example_number}.pth')
    print(f"Epoch {epoch+1}: Loss = {sum(loss_list)/len(loss_list):.4f}, Train Acc = {train_acc/len(X_train)*100:.4f}, Test Accuracy = {test_acc/len(X_test)*100:4f}")
