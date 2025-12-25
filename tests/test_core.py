import pytest
import numpy as np
import cirq
from protein_folding_vqe.interaction_models import hydrophobicity, electrostatic_potential, InteractionHamiltonian
from protein_folding_vqe.quantum_circuits import VariationalAnsatz
from protein_folding_vqe.vqe_optimizer import VQEOptimizer

def test_hydrophobicity():
    assert hydrophobicity('A') == 1.8
    assert hydrophobicity('R') == -4.5
    assert hydrophobicity('unknown') == 0.0

def test_electrostatic_potential():
    assert electrostatic_potential('K') == 1
    assert electrostatic_potential('D') == -1
    assert electrostatic_potential('A') == 0

def test_ansatz_creation():
    num_qubits = 4
    num_layers = 1
    ansatz = VariationalAnsatz(num_qubits, num_layers)
    assert len(ansatz.qubits) == num_qubits
    assert ansatz.num_params == num_layers * 2 * num_qubits
    
    params = np.zeros(ansatz.num_params)
    circuit = ansatz.build_circuit(params)
    assert isinstance(circuit, cirq.Circuit)

def test_vqe_optimizer_initialization():
    sequence = "ACD"
    optimizer = VQEOptimizer(sequence)
    assert optimizer.sequence == "ACD"
    assert optimizer.num_qubits == len(sequence) * 2

def test_compute_expectation():
    sequence = "A"
    optimizer = VQEOptimizer(sequence, num_qubits=2)
    params = np.zeros(optimizer.ansatz.num_params)
    energy = optimizer.compute_expectation(params)
    assert isinstance(energy, float)
