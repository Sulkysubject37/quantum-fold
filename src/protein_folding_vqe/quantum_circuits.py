"""Quantum circuit construction for VQE."""

import cirq
import numpy as np

def create_variational_circuit(qubits, params, layers=1):
    """
    Create a variational circuit with rotational and entanglement layers.
    """
    circuit = cirq.Circuit()
    num_qubits = len(qubits)
    
    # Calculate params per layer
    # 2 params per qubit (Rx, Rz)
    params_per_layer = 2 * num_qubits
    
    for layer in range(layers):
        layer_params = params[layer * params_per_layer : (layer + 1) * params_per_layer]
        
        # Single-qubit rotations
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(layer_params[i])(qubit))
            circuit.append(cirq.rz(layer_params[num_qubits + i])(qubit))
        
        # Entanglement: Ring topology (Nearest Neighbor)
        # This is more hardware-efficient than full connectivity
        for i in range(num_qubits):
            circuit.append(cirq.CZ(qubits[i], qubits[(i + 1) % num_qubits]))
    
    # Measurement layer is handled by the optimizer/sampler
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
        if len(params) != self.num_params:
            raise ValueError(f"Expected {self.num_params} params, got {len(params)}")
        return create_variational_circuit(self.qubits, params, self.num_layers)
    
    def get_initial_parameters(self, method='random'):
        """Generate initial parameters for optimization."""
        if method == 'random':
            return np.random.uniform(0, 2 * np.pi, self.num_params)
        elif method == 'zeros':
            return np.zeros(self.num_params)
        else:
            raise ValueError(f"Unknown parameter initialization method: {method}")
