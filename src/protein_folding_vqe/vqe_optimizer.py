"""VQE optimization implementation for 3D Lattice Folding."""

import numpy as np
from scipy.optimize import minimize
import cirq
from typing import List, Tuple, Callable, Optional, Dict
from .quantum_circuits import VariationalAnsatz
from .interaction_models import LatticeHamiltonian, InteractionHamiltonian

class VQEOptimizer:
    """Legacy VQEOptimizer (Kept for backward compatibility)."""
    def __init__(self, sequence: str, num_qubits: Optional[int] = None, num_layers: int = 1):
        self.sequence = sequence.upper()
        self.num_qubits = num_qubits or len(sequence) * 2
        self.num_layers = num_layers
        self.ansatz = VariationalAnsatz(self.num_qubits, num_layers)
        self.hamiltonian = InteractionHamiltonian(sequence)
        self.simulator = cirq.Simulator()
        
    def compute_expectation(self, params: np.ndarray) -> float:
        circuit = self.ansatz.build_circuit(params)
        result = self.simulator.simulate(circuit)
        state_vector = result.final_state_vector
        hamiltonian_terms = self.hamiltonian.build_hamiltonian_terms(self.ansatz.qubits)
        energy = 0.0
        qubit_map = {q: i for i, q in enumerate(self.ansatz.qubits)}
        for weight, pauli_string in hamiltonian_terms:
            exp_val = pauli_string.expectation_from_state_vector(state_vector, qubit_map=qubit_map)
            energy += weight * np.real(exp_val)
        return energy
    
    def optimize(self, initial_params: Optional[np.ndarray] = None) -> dict:
        if initial_params is None: initial_params = self.ansatz.get_initial_parameters()
        result = minimize(self.compute_expectation, initial_params, method='COBYLA')
        return {'optimal_energy': result.fun, 'success': result.success, 'message': result.message}


class LatticeVQEOptimizer:
    """
    Sampling-based VQE for 3D Protein Folding.
    Uses the Quantum Computer as a Generative Model for folding pathways.
    """
    
    def __init__(self, sequence: str, num_layers: int = 1, samples: int = 1000):
        self.sequence = sequence.upper()
        # We need 3 qubits per move. A protein of length N has N-1 moves.
        self.num_moves = len(sequence) - 1
        self.qubits_per_move = 3
        self.num_qubits = self.num_moves * self.qubits_per_move
        
        self.num_layers = num_layers
        self.samples = samples
        
        self.ansatz = VariationalAnsatz(self.num_qubits, num_layers)
        self.hamiltonian = LatticeHamiltonian(sequence)
        self.simulator = cirq.Simulator()
        
    def decode_bitstring(self, bits: List[int]) -> List[int]:
        """Convert a list of bits into a list of moves (0-7)."""
        moves = []
        for i in range(0, len(bits), self.qubits_per_move):
            chunk = bits[i : i + self.qubits_per_move]
            # Convert binary list [1, 0, 1] to integer 5
            move_val = 0
            for bit in chunk:
                move_val = (move_val << 1) | bit
            moves.append(move_val)
        return moves

    def compute_cost(self, params: np.ndarray) -> float:
        """
        Compute the expected energy of the quantum state.
        Cost = Average(Energy(sample) for sample in samples)
        """
        # 1. Build Circuit
        circuit = self.ansatz.build_circuit(params)
        
        # 2. Add Measurements
        circuit.append(cirq.measure(*self.ansatz.qubits, key='m'))
        
        # 3. Sample
        result = self.simulator.run(circuit, repetitions=self.samples)
        measurements = result.measurements['m'] # Shape: (samples, num_qubits)
        
        # 4. Calculate Average Energy
        total_energy = 0.0
        
        for sample_bits in measurements:
            moves = self.decode_bitstring(sample_bits)
            energy = self.hamiltonian.calculate_energy(moves)
            total_energy += energy
            
        avg_energy = total_energy / self.samples
        return avg_energy

    def optimize(self, initial_params: Optional[np.ndarray] = None, 
                 method: str = 'COBYLA', max_iter: int = 100) -> dict:
        """Run the classical optimization loop."""
        
        if initial_params is None:
            initial_params = self.ansatz.get_initial_parameters()
            
        # Callback to track progress
        history = []
        def callback(x):
            history.append(self.compute_cost(x))
            
        print(f"Optimizing 3D structure for sequence: {self.sequence}")
        print(f"Qubits: {self.num_qubits} | Parameters: {len(initial_params)}")
        
        result = minimize(
            self.compute_cost,
            initial_params,
            method=method,
            options={'maxiter': max_iter},
            callback=callback
        )
        
        # Final reconstruction
        # We run one last batch to find the absolute best sample seen
        best_structure_energy = float('inf')
        best_moves = None
        
        circuit = self.ansatz.build_circuit(result.x)
        circuit.append(cirq.measure(*self.ansatz.qubits, key='m'))
        final_results = self.simulator.run(circuit, repetitions=self.samples)
        
        for sample_bits in final_results.measurements['m']:
            moves = self.decode_bitstring(sample_bits)
            energy = self.hamiltonian.calculate_energy(moves)
            if energy < best_structure_energy:
                best_structure_energy = energy
                best_moves = moves
        
        return {
            'optimal_energy': best_structure_energy, # Best single sample
            'expected_energy': result.fun,           # Average of distribution
            'optimal_parameters': result.x,
            'best_moves': best_moves,
            'success': result.success,
            'message': result.message,
            'history': history
        }

def run_vqe_pipeline(sequence: str, num_layers: int = 1, mode: str = 'lattice') -> dict:
    """
    Run the VQE pipeline.
    
    Args:
        sequence: Amino acid sequence.
        mode: 'lattice' (3D) or 'legacy' (1D).
    """
    if mode == 'legacy':
        optimizer = VQEOptimizer(sequence, num_layers=num_layers)
        return optimizer.optimize()
    else:
        optimizer = LatticeVQEOptimizer(sequence, num_layers=num_layers)
        return optimizer.optimize()
