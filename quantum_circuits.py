"""Quantum circuit construction for VQE."""

import cirq
import numpy as np

def create_variational_circuit(qubits, params, layers=1):
    """
    Create a variational circuit with rotational and entanglement layers.
    
    Args:
        qubits: List of cirq.Qubit objects
        params: Array of parameters for rotational gates
        layers: Number of variational layers
    
    Returns:
        cirq.Circuit: Parameterized quantum circuit
    """
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    
    for layer in range(layers):
        # Single-qubit rotations
        for i, qubit in enumerate(qubits):
            param_idx = layer * 2 * num_qubits + i
            if param_idx < len(params):
                circuit.append(cirq.rx(params[param_idx])(qubit))
            
            param_idx = layer * 2 * num_qubits + num_qubits + i
            if param_idx < len(params):
                circuit.append(cirq.rz(params[param_idx])(qubit))
        
        # Entanglement layer
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                circuit.append(cirq.CZ(qubits[i], qubits[j]))
    
    return circuit

class VariationalAnsatz:
    """Flexible variational ansatz for protein folding."""
    
    def __init__(self, num_qubits, num_layers=1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qubits = cirq.LineQubit.range(num_qubits)
        self.num_params = num_layers * 2 * num_qubits
        
    def build_circuit(self, params):
        """Build circuit with given parameters."""
        return create_variational_circuit(self.qubits, params, self.num_layers)
    
    def get_initial_parameters(self, method='random'):
        """Generate initial parameters for optimization."""
        if method == 'random':
            return np.random.uniform(0, 2 * np.pi, self.num_params)
        elif method == 'zeros':
            return np.zeros(self.num_params)
        else:
            raise ValueError(f"Unknown parameter initialization method: {method}")