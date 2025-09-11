"""Error mitigation techniques for VQE protein folding."""

import cirq
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
import warnings
from scipy.optimize import curve_fit
from collections import defaultdict
import matplotlib.pyplot as plt

class ErrorMitigation:
    """Collection of error mitigation techniques for quantum circuits."""
    
    def __init__(self, simulator: Optional[cirq.Simulator] = None):
        """
        Initialize error mitigation techniques.
        
        Args:
            simulator: Cirq simulator instance (optional)
        """
        self.simulator = simulator or cirq.Simulator()
        self.mitigation_results = {}
    
    def zero_noise_extrapolation(self, 
                                circuit: cirq.Circuit, 
                                scale_factors: List[float] = [1.0, 2.0, 3.0],
                                noise_model: Optional[cirq.NoiseModel] = None,
                                num_shots: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """
        Zero Noise Extrapolation (ZNE) for error mitigation.
        
        Args:
            circuit: Quantum circuit to mitigate
            scale_factors: List of noise scaling factors
            noise_model: Noise model to apply
            num_shots: Number of shots for each scaling factor
            
        Returns:
            Tuple of (mitigated_expectation, results_dict)
        """
        if noise_model is None:
            noise_model = self._create_default_noise_model()
        
        expectations = []
        variances = []
        
        for scale_factor in scale_factors:
            scaled_circuit = self._scale_circuit_noise(circuit, scale_factor, noise_model)
            expectation, variance = self._measure_expectation(scaled_circuit, num_shots)
            expectations.append(expectation)
            variances.append(variance)
        
        # Fit to exponential decay or linear extrapolation
        try:
            # Try exponential fit first
            popt, _ = curve_fit(self._exponential_decay, scale_factors, expectations)
            mitigated_expectation = popt[0]  # Extrapolated to zero noise
            fit_type = 'exponential'
        except:
            # Fall back to linear extrapolation
            coefficients = np.polyfit(scale_factors, expectations, 1)
            mitigated_expectation = coefficients[1]  # Intercept at scale=0
            fit_type = 'linear'
        
        results = {
            'scale_factors': scale_factors,
            'expectations': expectations,
            'variances': variances,
            'mitigated_expectation': mitigated_expectation,
            'fit_type': fit_type,
            'raw_expectation': expectations[0]  # Unmitigated result
        }
        
        self.mitigation_results['zne'] = results
        return mitigated_expectation, results
    
    def randomized_compiling(self, 
                           circuit: cirq.Circuit, 
                           num_compilations: int = 10,
                           num_shots: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """
        Randomized Compiling (Twirling) for error mitigation.
        
        Args:
            circuit: Quantum circuit to mitigate
            num_compilations: Number of random compilations
            num_shots: Number of shots per compilation
            
        Returns:
            Tuple of (mitigated_expectation, results_dict)
        """
        compiled_expectations = []
        compiled_variances = []
        
        for _ in range(num_compilations):
            compiled_circuit = self._random_compile(circuit)
            expectation, variance = self._measure_expectation(compiled_circuit, num_shots)
            compiled_expectations.append(expectation)
            compiled_variances.append(variance)
        
        # Average over compilations
        mitigated_expectation = np.mean(compiled_expectations)
        mitigated_variance = np.mean(compiled_variances) / num_compilations
        
        results = {
            'num_compilations': num_compilations,
            'expectations': compiled_expectations,
            'variances': compiled_variances,
            'mitigated_expectation': mitigated_expectation,
            'mitigated_variance': mitigated_variance,
            'std_dev': np.std(compiled_expectations)
        }
        
        self.mitigation_results['randomized_compiling'] = results
        return mitigated_expectation, results
    
    def measurement_error_mitigation(self, 
                                   circuit: cirq.Circuit,
                                   calibration_shots: int = 10000,
                                   num_shots: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """
        Measurement Error Mitigation using calibration matrix.
        
        Args:
            circuit: Quantum circuit to mitigate
            calibration_shots: Shots for calibration matrix
            num_shots: Number of shots for actual measurement
            
        Returns:
            Tuple of (mitigated_expectation, results_dict)
        """
        # Create calibration circuits for each computational basis state
        num_qubits = len(list(circuit.all_qubits()))
        calibration_matrix = self._build_calibration_matrix(num_qubits, calibration_shots)
        
        # Measure raw probabilities
        raw_results = self.simulator.run(circuit, repetitions=num_shots)
        raw_counts = raw_results.histogram(key='m')
        
        # Apply mitigation
        mitigated_probs = self._apply_measurement_mitigation(raw_counts, calibration_matrix, num_qubits)
        
        # Calculate expectation value from mitigated probabilities
        mitigated_expectation = self._expectation_from_probs(mitigated_probs, num_qubits)
        
        results = {
            'calibration_matrix': calibration_matrix,
            'raw_counts': raw_counts,
            'mitigated_probs': mitigated_probs,
            'mitigated_expectation': mitigated_expectation,
            'num_qubits': num_qubits
        }
        
        self.mitigation_results['measurement_mitigation'] = results
        return mitigated_expectation, results
    
    def dynamical_decoupling(self, 
                           circuit: cirq.Circuit,
                           dd_sequence: List[cirq.Gate] = None,
                           num_shots: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """
        Dynamical Decoupling for coherence time extension.
        
        Args:
            circuit: Quantum circuit to mitigate
            dd_sequence: Dynamical decoupling sequence
            num_shots: Number of shots
            
        Returns:
            Tuple of (mitigated_expectation, results_dict)
        """
        if dd_sequence is None:
            dd_sequence = [cirq.X, cirq.X]  # Basic XYXY sequence
        
        protected_circuit = self._insert_dd_sequences(circuit, dd_sequence)
        expectation, variance = self._measure_expectation(protected_circuit, num_shots)
        
        results = {
            'dd_sequence': [str(gate) for gate in dd_sequence],
            'protected_expectation': expectation,
            'variance': variance,
            'circuit_depth_increase': len(protected_circuit) - len(circuit)
        }
        
        self.mitigation_results['dynamical_decoupling'] = results
        return expectation, results
    
    def combined_mitigation(self, 
                          circuit: cirq.Circuit,
                          techniques: List[str] = ['zne', 'randomized_compiling', 'measurement_mitigation'],
                          **kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Combine multiple error mitigation techniques.
        
        Args:
            circuit: Quantum circuit to mitigate
            techniques: List of mitigation techniques to apply
            **kwargs: Additional arguments for specific techniques
            
        Returns:
            Tuple of (mitigated_expectation, comprehensive_results_dict)
        """
        results = {}
        current_circuit = circuit
        current_expectation = None
        
        for technique in techniques:
            if technique == 'randomized_compiling':
                expectation, tech_results = self.randomized_compiling(
                    current_circuit, 
                    kwargs.get('num_compilations', 10),
                    kwargs.get('num_shots', 1000)
                )
            elif technique == 'measurement_mitigation':
                expectation, tech_results = self.measurement_error_mitigation(
                    current_circuit,
                    kwargs.get('calibration_shots', 10000),
                    kwargs.get('num_shots', 1000)
                )
            elif technique == 'zne':
                expectation, tech_results = self.zero_noise_extrapolation(
                    current_circuit,
                    kwargs.get('scale_factors', [1.0, 2.0, 3.0]),
                    kwargs.get('noise_model', None),
                    kwargs.get('num_shots', 1000)
                )
            elif technique == 'dynamical_decoupling':
                expectation, tech_results = self.dynamical_decoupling(
                    current_circuit,
                    kwargs.get('dd_sequence', None),
                    kwargs.get('num_shots', 1000)
                )
            else:
                warnings.warn(f"Unknown mitigation technique: {technique}")
                continue
            
            results[technique] = tech_results
            current_expectation = expectation
        
        combined_results = {
            'techniques_applied': techniques,
            'final_expectation': current_expectation,
            'technique_results': results
        }
        
        self.mitigation_results['combined'] = combined_results
        return current_expectation, combined_results
    
    def _create_default_noise_model(self) -> cirq.NoiseModel:
        """Create a default noise model for simulation."""
        return cirq.ConstantQubitNoiseModel(
            depolarize=0.01,  # 1% depolarizing noise
            amplitude_damp=0.005,  # 0.5% amplitude damping
            phase_damp=0.005  # 0.5% phase damping
        )
    
    def _scale_circuit_noise(self, 
                           circuit: cirq.Circuit, 
                           scale_factor: float,
                           noise_model: cirq.NoiseModel) -> cirq.Circuit:
        """Scale circuit noise by inserting identity gates."""
        # Simple implementation: insert identity gates to scale depth
        scaled_circuit = circuit.copy()
        qubits = list(circuit.all_qubits())
        
        for _ in range(int(scale_factor) - 1):
            for qubit in qubits:
                scaled_circuit.append(cirq.I(qubit))
        
        return scaled_circuit
    
    def _measure_expectation(self, circuit: cirq.Circuit, num_shots: int) -> Tuple[float, float]:
        """Measure expectation value and variance."""
        # Add measurement if not present
        if not any(op.gate == cirq.measure for op in circuit.all_operations()):
            qubits = list(circuit.all_qubits())
            circuit.append(cirq.measure(*qubits, key='m'))
        
        results = self.simulator.run(circuit, repetitions=num_shots)
        counts = results.histogram(key='m')
        
        # Calculate expectation value (simplified)
        total = sum(counts.values())
        expectation = sum(state * count for state, count in counts.items()) / total
        variance = sum((state - expectation)**2 * count for state, count in counts.items()) / total
        
        return float(expectation), float(variance)
    
    def _random_compile(self, circuit: cirq.Circuit) -> cirq.Circuit:
        """Randomly compile circuit by inserting random Pauli gates."""
        compiled_circuit = circuit.copy()
        qubits = list(circuit.all_qubits())
        
        # Insert random Pauli gates at random positions
        num_insertions = np.random.randint(1, len(qubits))
        for _ in range(num_insertions):
            qubit = np.random.choice(qubits)
            pauli_gate = np.random.choice([cirq.X, cirq.Y, cirq.Z])
            position = np.random.randint(0, len(compiled_circuit))
            
            # Insert at random position
            compiled_circuit._moments.insert(position, cirq.Moment([pauli_gate(qubit)]))
        
        return compiled_circuit
    
    def _build_calibration_matrix(self, num_qubits: int, shots: int) -> np.ndarray:
        """Build measurement calibration matrix."""
        matrix = np.zeros((2**num_qubits, 2**num_qubits))
        
        for prepared_state in range(2**num_qubits):
            # Create circuit to prepare computational basis state
            prep_circuit = cirq.Circuit()
            qubits = cirq.LineQubit.range(num_qubits)
            
            # Prepare basis state
            for i, qubit in enumerate(qubits):
                if (prepared_state >> i) & 1:
                    prep_circuit.append(cirq.X(qubit))
            
            # Measure
            prep_circuit.append(cirq.measure(*qubits, key='m'))
            
            # Run and get counts
            results = self.simulator.run(prep_circuit, repetitions=shots)
            counts = results.histogram(key='m')
            
            # Fill calibration matrix
            for measured_state, count in counts.items():
                matrix[prepared_state, measured_state] = count / shots
        
        return matrix
    
    def _apply_measurement_mitigation(self, counts: Dict[int, int], 
                                    calibration_matrix: np.ndarray, 
                                    num_qubits: int) -> np.ndarray:
        """Apply measurement error mitigation."""
        total_shots = sum(counts.values())
        raw_probs = np.zeros(2**num_qubits)
        
        for state, count in counts.items():
            raw_probs[state] = count / total_shots
        
        # Solve: calibration_matrix @ mitigated_probs = raw_probs
        mitigated_probs = np.linalg.lstsq(calibration_matrix, raw_probs, rcond=None)[0]
        mitigated_probs = np.clip(mitigated_probs, 0, 1)
        mitigated_probs /= np.sum(mitigated_probs)  # Renormalize
        
        return mitigated_probs
    
    def _expectation_from_probs(self, probs: np.ndarray, num_qubits: int) -> float:
        """Calculate expectation value from probability distribution."""
        expectation = 0.0
        for state in range(len(probs)):
            # Convert state to expectation value (simplified)
            expectation += state * probs[state]
        return expectation
    
    def _insert_dd_sequences(self, circuit: cirq.Circuit, dd_sequence: List[cirq.Gate]) -> cirq.Circuit:
        """Insert dynamical decoupling sequences into circuit."""
        protected_circuit = circuit.copy()
        qubits = list(circuit.all_qubits())
        
        # Find idle moments and insert DD sequences
        for moment_idx, moment in enumerate(circuit):
            if len(moment.operations) < len(qubits):
                # Some qubits are idle in this moment
                idle_qubits = set(qubits) - {op.qubits[0] for op in moment.operations}
                
                for qubit in idle_qubits:
                    for dd_gate in dd_sequence:
                        protected_circuit._moments[moment_idx] = protected_circuit._moments[moment_idx].with_operation(
                            dd_gate(qubit)
                        )
        
        return protected_circuit
    
    def _exponential_decay(self, x: float, a: float, b: float, c: float) -> float:
        """Exponential decay function for ZNE fitting."""
        return a * np.exp(-b * x) + c
    
    def plot_zne_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Zero Noise Extrapolation results."""
        if 'zne' not in self.mitigation_results:
            raise ValueError("ZNE results not available")
        
        results = self.mitigation_results['zne']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(
            results['scale_factors'], 
            results['expectations'],
            yerr=np.sqrt(results['variances']),
            fmt='o-', 
            label='Measured'
        )
        
        # Plot extrapolation
        x_fit = np.linspace(0, max(results['scale_factors']) * 1.1, 100)
        if results['fit_type'] == 'exponential':
            y_fit = self._exponential_decay(x_fit, *np.polyfit(results['scale_factors'], results['expectations'], 2)[:3])
        else:
            y_fit = np.poly1d(np.polyfit(results['scale_factors'], results['expectations'], 1))(x_fit)
        
        ax.plot(x_fit, y_fit, 'r--', label='Extrapolation')
        ax.axhline(y=results['mitigated_expectation'], color='g', linestyle='--', label='Mitigated')
        ax.axhline(y=results['raw_expectation'], color='b', linestyle='--', label='Raw')
        
        ax.set_xlabel('Noise Scale Factor')
        ax.set_ylabel('Expectation Value')
        ax.set_title('Zero Noise Extrapolation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Utility functions for protein folding specific error mitigation
def create_protein_folding_noise_model(sequence_length: int) -> cirq.NoiseModel:
    """
    Create a noise model tailored for protein folding circuits.
    
    Args:
        sequence_length: Length of protein sequence
        
    Returns:
        Custom noise model
    """
    # Scale noise with circuit complexity
    complexity_factor = min(1.0, sequence_length / 10.0)  # Cap at 1.0
    
    return cirq.ConstantQubitNoiseModel(
        depolarize=0.005 + 0.015 * complexity_factor,
        amplitude_damp=0.002 + 0.008 * complexity_factor,
        phase_damp=0.002 + 0.008 * complexity_factor
    )

def mitigate_vqe_expectation(
    circuit: cirq