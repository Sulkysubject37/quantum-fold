"""Utility functions for protein folding VQE."""

import numpy as np
import cirq
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def validate_amino_acid_sequence(sequence: str) -> bool:
    """
    Validate if the input sequence contains valid amino acid codes.
    
    Args:
        sequence: Protein amino acid sequence
        
    Returns:
        bool: True if valid, False otherwise
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence_upper = sequence.upper()
    
    if not sequence_upper:
        raise ValueError("Sequence cannot be empty")
    
    invalid_chars = set(sequence_upper) - valid_amino_acids
    if invalid_chars:
        raise ValueError(f"Invalid amino acid characters: {invalid_chars}")
    
    return True

def sequence_to_properties(sequence: str) -> Dict[str, List[float]]:
    """
    Convert amino acid sequence to property vectors.
    
    Args:
        sequence: Protein amino acid sequence
        
    Returns:
        Dictionary with property vectors
    """
    from .interaction_models import hydrophobicity, electrostatic_potential
    
    sequence_upper = sequence.upper()
    validate_amino_acid_sequence(sequence_upper)
    
    return {
        'hydrophobicity': [hydrophobicity(aa) for aa in sequence_upper],
        'electrostatic': [electrostatic_potential(aa) for aa in sequence_upper],
        'sequence': list(sequence_upper)
    }

def calculate_sequence_complexity(sequence: str) -> float:
    """
    Calculate a complexity score for the protein sequence.
    
    Args:
        sequence: Protein amino acid sequence
        
    Returns:
        float: Complexity score (0-1)
    """
    from collections import Counter
    
    sequence_upper = sequence.upper()
    total_length = len(sequence_upper)
    
    if total_length == 0:
        return 0.0
    
    # Count unique amino acids
    unique_count = len(set(sequence_upper))
    
    # Calculate Shannon entropy
    counts = Counter(sequence_upper)
    entropy = -sum((count / total_length) * np.log2(count / total_length) 
                  for count in counts.values())
    
    # Normalize complexity score
    max_entropy = np.log2(min(20, total_length))  # 20 standard amino acids
    complexity = (unique_count / 20) * 0.5 + (entropy / max_entropy) * 0.5
    
    return min(complexity, 1.0)

def estimate_required_qubits(sequence: str, qubits_per_aa: int = 2) -> int:
    """
    Estimate number of qubits required for the sequence.
    
    Args:
        sequence: Protein amino acid sequence
        qubits_per_aa: Qubits per amino acid
        
    Returns:
        int: Estimated number of qubits
    """
    return len(sequence) * qubits_per_aa

def save_results(results: Dict[str, Any], filename: str, format: str = 'json') -> None:
    """
    Save VQE results to file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        format: File format ('json', 'pkl')
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, int)):
                json_results[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                json_results[key] = float(value)
            elif isinstance(value, list):
                 # Handle list of numpy types
                 json_results[key] = [int(x) if isinstance(x, (np.integer, int)) else float(x) if isinstance(x, (np.floating, float)) else x for x in value]
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif format == 'pkl':
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_results(filename: str, format: str = 'json') -> Dict[str, Any]:
    """
    Load VQE results from file.
    
    Args:
        filename: Input filename
        format: File format ('json', 'pkl')
        
    Returns:
        Dictionary with results
    """
    if format == 'json':
        with open(filename, 'r') as f:
            return json.load(f)
    
    elif format == 'pkl':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def plot_energy_convergence(energy_history: List[float], title: str = "VQE Energy Convergence") -> plt.Figure:
    """
    Plot VQE energy convergence.
    
    Args:
        energy_history: List of energy values during optimization
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energy_history, 'b-', linewidth=2, label='Energy')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def generate_benchmark_report(results: Dict[str, Any]) -> str:
    """
    Generate a benchmark report from VQE results.
    
    Args:
        results: VQE results dictionary
        
    Returns:
        str: Formatted benchmark report
    """
    report = [
        "VQE Benchmark Report",
        "====================",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Sequence: {results.get('sequence', 'N/A')}",
        f"Sequence length: {len(results.get('sequence', ''))}",
        f"Number of qubits: {results.get('num_qubits', 'N/A')}",
        f"Number of layers: {results.get('num_layers', 'N/A')}",
        f"Optimal energy: {results.get('optimal_energy', 'N/A'):.6f}",
        f"Optimization success: {results.get('success', 'N/A')}",
        f"Message: {results.get('message', 'N/A')}",
        f"Complexity score: {calculate_sequence_complexity(results.get('sequence', '')):.3f}"
    ]
    
    return "\n".join(report)

def check_cirq_availability() -> Tuple[bool, str]:
    """
    Check if Cirq is available and which backends are accessible.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        import cirq
        import cirq_google
        
        message = f"Cirq {cirq.__version__} available. "
        
        # Check simulator availability
        simulator = cirq.Simulator()
        message += "Simulator: ✓. "
        
        # Check if Google Quantum Computing Service is available
        try:
            # This will only work if proper authentication is set up
            import cirq_google as cg
            message += "Google Quantum Service: Authentication required."
        except:
            message += "Google Quantum Service: Not configured."
        
        return True, message
        
    except ImportError as e:
        return False, f"Cirq not available: {str(e)}"

def create_timestamped_filename(prefix: str, extension: str = "json") -> str:
    """
    Create a timestamped filename for results.
    
    Args:
        prefix: Filename prefix
        extension: File extension
        
    Returns:
        str: Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def normalize_parameters(params: np.ndarray) -> np.ndarray:
    """
    Normalize parameters to [-π, π] range.
    
    Args:
        params: Parameter array
        
    Returns:
        Normalized parameter array
    """
    return (params + np.pi) % (2 * np.pi) - np.pi

def calculate_parameter_variance(optimal_params: np.ndarray) -> float:
    """
    Calculate variance of optimal parameters.
    
    Args:
        optimal_params: Optimal parameter array
        
    Returns:
        float: Parameter variance
    """
    return float(np.var(optimal_params))