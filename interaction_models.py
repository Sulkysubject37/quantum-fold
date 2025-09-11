"""Amino acid interaction models for protein folding."""

import numpy as np

def hydrophobicity(amino_acid: str) -> float:
    """Return hydrophobicity value for an amino acid."""
    hydrophobic = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    return hydrophobic.get(amino_acid.upper(), 0.0)

def electrostatic_potential(amino_acid: str) -> float:
    """Return electrostatic potential based on amino acid charge."""
    charge = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    return charge.get(amino_acid.upper(), 0.0)

def van_der_waals_interaction(aa1: str, aa2: str) -> float:
    """Calculate van der Waals interaction between two amino acids."""
    # Simplified model based on hydrophobicity difference
    h1 = hydrophobicity(aa1)
    h2 = hydrophobicity(aa2)
    return 0.1 * abs(h1 - h2)

class InteractionHamiltonian:
    """Build Hamiltonian for protein folding interactions."""
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.qubits = None
        
    def build_hamiltonian_terms(self, qubits):
        """Build Hamiltonian terms for all interactions."""
        self.qubits = qubits
        hamiltonian_terms = []
        
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                aa_i = self.sequence[i % len(self.sequence)]
                aa_j = self.sequence[j % len(self.sequence)]
                
                # Hydrophobic interaction (ZZ term)
                hydro_weight = hydrophobicity(aa_i) * hydrophobicity(aa_j)
                zz_term = cirq.PauliString({qubits[i]: cirq.Z, qubits[j]: cirq.Z})
                hamiltonian_terms.append((hydro_weight, zz_term))
                
                # Electrostatic interaction
                elec_weight = 0.5 * electrostatic_potential(aa_i) * electrostatic_potential(aa_j)
                elec_term = cirq.PauliString({qubits[i]: cirq.Z, qubits[j]: cirq.Z})
                hamiltonian_terms.append((elec_weight, elec_term))
                
                # Van der Waals interaction
                vdw_weight = van_der_waals_interaction(aa_i, aa_j)
                vdw_term = cirq.PauliString({qubits[i]: cirq.Z, qubits[j]: cirq.Z})
                hamiltonian_terms.append((vdw_weight, vdw_term))
                
        return hamiltonian_terms