'''This file contains code that shows scalability of proposed multi hardware
   training methodology to Cifar-10 dataset. We specifically select 
   classes 0 and 6 for training.'''

import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.utils.data as Data
from sklearn.datasets import load_iris, load_digits, fetch_openml
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
config = input("Enter configuration(qubits:20/27/127, layout:I/II/III/IV/V, eg: '27 I'):")
config_list = config.split()
assert len(config_list) == 2 and (config_list[0] == '20' or config_list[0] == '27' or config_list[0] == '127')\
    and (config_list[1] == 'I' or config_list[1] == 'II' or config_list[1] == 'III' or config_list[1] == 'IV' or config_list[1] == 'V')\
    , "Configuration data not entered properly, please try again."
num_layers = 6
batch_size = 16

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, label_map):
        self.dataset = Subset(dataset, indices)
        self.label_map = label_map

    def __getitem__(self, index):
        x, y = self.dataset[index]
        # Remap the label
        return x, self.label_map[y]

    def __len__(self):
        return len(self.dataset)

# Load the CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Filter indices for classes 3 and 5
classes = [0,6]
indices = [i for i, (_, label) in enumerate(cifar10_train) if label in classes]

# Sample 150 indices from the filtered indices
selected_indices = np.random.choice(indices, 600, replace=False)

# Split indices for training and testing
train_indices = selected_indices[:420]  # 70% for training
test_indices = selected_indices[420:]  # 30% for testing

# Define label mapping (3 -> 0, 5 -> 1)
label_map = {}
for i in range(len(classes)):
    label_map[classes[i]] = i

# Create custom dataset instances for train and test
train_data = CustomDataset(cifar10_train, train_indices, label_map)
test_data = CustomDataset(cifar10_train, test_indices, label_map)

# Create DataLoaders for the train and test sets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


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
def pqc_cifar_strong(inputs, params):
    qml.templates.AngleEmbedding(inputs, wires=(range(num_qubits)))
    qml.templates.StronglyEntanglingLayers(params, wires=range(num_qubits), ranges=[1]*num_layers)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

expanded_cifar_circuit = qml.transforms.broadcast_expand(pqc_cifar_strong) # For extra processing

# Define the model
weight_shapes = {'params': (num_layers, num_qubits, 3)}
torch.manual_seed(61)
qlayer = qml.qnn.TorchLayer(pqc_cifar_strong, weight_shapes, init_method=nn.init.normal_)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qlayer
        self.conv1 = nn.Conv2d(3,4,3,2,1)
        self.conv2 = nn.Conv2d(4,4,3,2,1)
        self.conv3 = nn.Conv2d(4,4,3,2,1)
        self.conv4 = nn.Conv2d(4,8,3,4,1)
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        conv4_out = self.conv4(conv3_out)
        conv4_out_flattened = np.pi*torch.tanh(torch.flatten(conv4_out, 1))
        out = self.qlayer(conv4_out_flattened.to(device))
        return out

model = Model().to(device)

'''Use this only when model is already trained for few epochs once and you are training the same model'''
# model.load_state_dict(torch.load('strongly_entangling_layers_cifar_27hw_baseline_range1_hybrid_8q.pth'))

# Define the epochs, loss function and optimizer
epochs = 2
loss_fn = nn.CrossEntropyLoss()
soft_out = nn.Softmax(dim=1)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
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
        torch.save(model.state_dict(), 'strongly_entangling_layers_cifar_27hw_baseline_range1_hybrid_8q.pth')
    print(f"Epoch {epoch+1}: Loss = {sum(loss_list)/len(loss_list):.4f}, Train Acc = {train_acc/len(train_loader.dataset)*100:.2f}, Test Accuracy = {test_acc/len(test_loader.dataset)*100:2f}")
