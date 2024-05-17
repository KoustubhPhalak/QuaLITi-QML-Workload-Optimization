'''This file contains extra handy functions implemented, primarily code for obtaining
   subset backend from a large backend. We have done this to cut down simulation time
   significantly.'''

import numpy as np
import qiskit
from qiskit.circuit.library import (
    IGate, CXGate, RZGate, SXGate,
    XGate, U1Gate, U2Gate, U3Gate,
    Reset, Measure, CZGate
)
from qiskit.circuit import Delay
from qiskit_aer.noise import NoiseModel
import json
import inspect
from qiskit.providers.fake_provider.fake_backend import FakeBackendV2
import qiskit_aer.noise as noise
from qiskit_aer import AerSimulator, QasmSimulator
from qiskit.transpiler import CouplingMap
from qiskit.providers.models import BackendConfiguration
from qiskit.providers.fake_provider import Fake127QPulseV1, Fake27QPulseV1, Fake5QV1, Fake20QV1, Fake7QPulseV1

def qubit_numbering_mapping(layout_qubits, layout_coupling):
    num_qubits = len(layout_qubits)
    qubit_numbering_dict = {}
    for cnt in range(num_qubits):
        qubit_numbering_dict[layout_qubits[cnt]]=cnt
    new_layout_qubits = list(range(num_qubits))
    new_layout_coupling = []
    for coupling in layout_coupling:
        new_layout_coupling.append([qubit_numbering_dict[coupling[0]], qubit_numbering_dict[coupling[1]]])
    return new_layout_qubits, new_layout_coupling

def create_subset_backend(layout_qubits, layout_coupling, backend):
    noise_model = NoiseModel.from_backend(backend)
    mro = inspect.getmro(type(backend))
    flag = FakeBackendV2 in mro
    new_layout_qubits, new_layout_coupling = qubit_numbering_mapping(layout_qubits, layout_coupling)

    # backend_cm = backend.configuration().coupling_map
    # backend_2q_noise_dict = {}
    # layout_coupling = backend_cm

    basis_gates = noise_model.basis_gates
    gate_map = {'cx':CXGate(), 'id':IGate(), 
                'u1':U1Gate(0), 'u2':U2Gate(0,0), 
                'u3':U3Gate(0,0,0), 'x':XGate(), 
                'sx':SXGate(), 'rz':RZGate(0), 
                'reset':Reset(), 'delay':Delay(1),
                'measure':Measure(), 'cz':CZGate()}
    single_qubit_basis_gates = [gate for gate in basis_gates if gate_map[gate].num_qubits == 1]
    two_qubit_basis_gates = [gate for gate in basis_gates if gate_map[gate].num_qubits == 2]

    if flag == False: # if backend has base class FakeBackendV1
        single_qubit_error_rates = [[] for _ in range(len(layout_qubits))]
        for i in range(len(layout_qubits)):
            for gate in single_qubit_basis_gates:
                if gate == 'reset' or gate == 'measure' or gate == 'delay':
                    continue
                try:
                    single_qubit_error_rates[i].append(backend.properties().gate_error(gate, layout_qubits[i]))
                except:
                    single_qubit_error_rates[i].append(0)

        cx_error_rates = [[] for _ in range(len(layout_coupling))]
        for i in range(len(layout_coupling)):
            for gate in two_qubit_basis_gates:
                try:
                    cx_error_rates[i].append(backend.properties().gate_error(gate, layout_coupling[i]))
                except:
                    cx_error_rates[i].append(0)

        single_qubit_error_rates = [max(single_qubit_error_rates[i]) for i in range(len(layout_qubits))]
        cx_error_rates = [max(cx_error_rates[i]) for i in range(len(layout_coupling))]
        # print(np.mean(cx_error_rates))
        # for i in range(len(layout_coupling)):
        #     backend_2q_noise_dict[tuple(layout_coupling[i])] = cx_error_rates[i]
        # backend_2q_noise_dict = sorted(backend_2q_noise_dict.items(), key=lambda x:x[1], reverse=True)
        # print(backend_2q_noise_dict)
        # exit()
        # for i in cx_error_rates:
        #     print(f"{i:.4f}",end="\t"),
        # print(f"{np.mean(cx_error_rates):.4f}")
        # exit()

        t1_times = [backend.properties().t1(qubit)/(1e-6) for qubit in layout_qubits]
        # t1_times = [round(elem, 0) for elem in t1_times]
        t2_times = [backend.properties().t2(qubit)/(1e-6) for qubit in layout_qubits]
        # t2_times = [round(elem, 0) for elem in t2_times]
        # print(f"t1_times:{t1_times} : {round(np.mean(t1_times),0)}\u00B1{round(np.std(t1_times),0)}\nt2_times:{t2_times} : {round(np.mean(t2_times),0)}\u00B1{round(np.std(t2_times),0)}")
        # exit()

        gate_times = [[] for _ in range(len(layout_qubits))]
        for i in range(len(layout_qubits)):
            for gate in single_qubit_basis_gates:
                if gate == 'reset' or gate == 'measure' or gate == 'delay':
                    continue
                try:
                    gate_times[i].append(backend.properties().gate_length(gate, layout_qubits[i]))
                except:
                    gate_times[i].append(0)
        gate_times = [max(gate_times[i])/(1e-6) for i in range(len(layout_qubits))]

        if 'reset' in basis_gates:
            reset_times = [backend.properties().gate_property('reset',qubit)['gate_length'][0]/(1e-6) for qubit in layout_qubits]
        readout_probabilities = [[backend.properties().qubit_property(qubit)['prob_meas1_prep0'][0], backend.properties().qubit_property(qubit)['prob_meas0_prep1'][0]] for qubit in layout_qubits]
        # readout_array = np.array(readout_probabilities)
        # mean_0_but_1 = np.mean(readout_array[:, 0])
        # mean_1_but_0 = np.mean(readout_array[:, 1])
        # print(readout_probabilities)
        # print(f"0 but 1: {mean_0_but_1:.3f}, 1 but 0: {mean_1_but_0:.3f}")
        # exit()
    elif flag == True: # if backend has base class FakeBackendV2
        f = open(f"{backend.dirname}/{backend.props_filename}")
        noise_backend_json = json.load(f)
        single_qubit_gate_error_rates = [[] for _ in range(len(layout_qubits))]
        for qubit in layout_qubits:
            for gate in single_qubit_basis_gates:
                if gate == 'reset' or gate == 'measure' or gate == 'delay':
                    continue
                for i in range(len(noise_backend_json['gates'])):
                    if noise_backend_json['gates'][i]['gate'] == gate and noise_backend_json['gates'][i]['qubits'] == [qubit]:
                        single_qubit_gate_error_rates[layout_qubits.index(qubit)].append(noise_backend_json['gates'][i]['parameters'][0]['value'])

        cx_error_rates = [[] for _ in range(len(layout_coupling))]
        for coupling in layout_coupling:
            for gate in two_qubit_basis_gates:
                if gate == 'reset' or gate == 'measure' or gate == 'delay':
                    continue
                for i in range(len(noise_backend_json['gates'])):
                    if noise_backend_json['gates'][i]['gate'] == gate and noise_backend_json['gates'][i]['qubits'] == coupling:
                        cx_error_rates[layout_coupling.index(coupling)].append(noise_backend_json['gates'][i]['parameters'][0]['value'])                        

        single_qubit_error_rates = [max(single_qubit_gate_error_rates[i]) for i in range(len(layout_qubits))]
        cx_error_rates = [max(cx_error_rates[i]) for i in range(len(layout_coupling))]
        t1_times = [noise_backend_json['qubits'][qubit][0]['value'] for qubit in layout_qubits]
        t2_times = [noise_backend_json['qubits'][qubit][1]['value'] for qubit in layout_qubits]

        gate_times = [[] for _ in range(len(layout_qubits))]
        for qubit in layout_qubits:
            for gate in single_qubit_basis_gates:
                if gate == 'reset' or gate == 'measure' or gate == 'delay':
                    continue
                for i in range(len(noise_backend_json['gates'])):
                    if noise_backend_json['gates'][i]['gate'] == gate and noise_backend_json['gates'][i]['qubits'] == [qubit]:
                        gate_times[layout_qubits.index(qubit)].append(noise_backend_json['gates'][i]['parameters'][1]['value']*(1e-3))
        gate_times = [max(gate_times[i]) for i in range(len(layout_qubits))]

        if 'reset' in basis_gates:
            reset_times = [noise_backend_json['gates'][i]['parameters'][0]['value']*(1e-3) for i in range(len(noise_backend_json['gates'])) if noise_backend_json['gates'][i]['gate'] == 'reset' and noise_backend_json['gates'][i]['qubits'][0] in layout_qubits]
        readout_probabilities = [[noise_backend_json['qubits'][qubit][6]['value'], noise_backend_json['qubits'][qubit][5]['value']] for qubit in layout_qubits]

    single_qubit_errors = [noise.depolarizing_error(rate, 1) for rate in single_qubit_error_rates]
    cx_errors = [noise.depolarizing_error(rate, 2) for rate in cx_error_rates] 
    decoherence_errors = [noise.thermal_relaxation_error(t1_times[i], t2_times[i], gate_times[i]) for i in range(len(layout_qubits))]
    if 'reset' in basis_gates:
        reset_decoherence_errors = [noise.thermal_relaxation_error(t1_times[i], t2_times[i], reset_times[i]) for i in range(len(layout_qubits))]
    readout_errors = [noise.ReadoutError([[1-readout_probabilities[i][0], readout_probabilities[i][0]],[readout_probabilities[i][1], 1-readout_probabilities[i][1]]]) for i in range(len(layout_qubits))]

    noise_model_partial = noise.NoiseModel(basis_gates=noise_model.basis_gates)
    for i in range(len(new_layout_qubits)):
        noise_model_partial.add_quantum_error(single_qubit_errors[i], list(set(single_qubit_basis_gates) - set(['reset'])), [new_layout_qubits[i]], warnings=False)
        noise_model_partial.add_quantum_error(decoherence_errors[i], list(set(single_qubit_basis_gates) - set(['reset'])), [new_layout_qubits[i]], warnings=False)
            
        if 'reset' in basis_gates:
            noise_model_partial.add_quantum_error(reset_decoherence_errors[i], ['reset'], [layout_qubits[i]], warnings=False)
        noise_model_partial.add_readout_error(readout_errors[i], [new_layout_qubits[i]])

    for i in range(len(layout_coupling)):
        noise_model_partial.add_quantum_error(cx_errors[i], two_qubit_basis_gates, new_layout_coupling[i])

    sim_config={"backend_name":"custom_subset_backend",
                "backend_version":"1.0",
                "n_qubits":len(new_layout_qubits),
                "basis_gates":noise_model_partial.basis_gates,
                "gates":[],
                "local":True,
                "simulator":True,
                "conditional":True,
                "open_pulse":True,
                "memory":True,
                "max_shots":1024,
                "coupling_map":new_layout_coupling,
                "description":"Subset backend for a larger fake backend based on inputed layout qubits and layout coupling",
                }
                
    backend_config = BackendConfiguration.from_dict(sim_config)
    custom_backend = AerSimulator(configuration=backend_config, noise_model=noise_model_partial)
    custom_backend.set_options(device='GPU', method='density_matrix', batched_shots_gpu=True,)
    return noise_model_partial, custom_backend