"""
Secondary Structure Propensity Predictor & Motif Templates.
Enforces strict geometric patterns for Helices and Sheets on the Lattice.
"""

from typing import Dict, List, Tuple
import numpy as np

# --- 1. Propensity Data (Simplified Chou-Fasman) ---
PROPENSITIES = {
    'A': {'P_a': 1.42, 'P_b': 0.83}, 
    'R': {'P_a': 0.98, 'P_b': 0.93},
    'N': {'P_a': 0.67, 'P_b': 0.89},
    'D': {'P_a': 1.01, 'P_b': 0.54},
    'C': {'P_a': 0.70, 'P_b': 1.19},
    'Q': {'P_a': 1.11, 'P_b': 1.10},
    'E': {'P_a': 1.51, 'P_b': 0.37}, 
    'G': {'P_a': 0.57, 'P_b': 0.75}, 
    'H': {'P_a': 1.00, 'P_b': 0.87},
    'I': {'P_a': 1.08, 'P_b': 1.60}, 
    'L': {'P_a': 1.21, 'P_b': 1.30},
    'K': {'P_a': 1.16, 'P_b': 0.74},
    'M': {'P_a': 1.45, 'P_b': 1.05},
    'F': {'P_a': 1.13, 'P_b': 1.38},
    'P': {'P_a': 0.57, 'P_b': 0.55}, 
    'S': {'P_a': 0.77, 'P_b': 0.75},
    'T': {'P_a': 0.83, 'P_b': 1.19},
    'W': {'P_a': 1.08, 'P_b': 1.37},
    'Y': {'P_a': 0.69, 'P_b': 1.47},
    'V': {'P_a': 1.06, 'P_b': 1.70}, 
}

def get_structure_propensity(sequence: str) -> List[str]:
    """
    Predict secondary structure ('H', 'E', 'C') for each residue.
    """
    structure = []
    # Smoothing window to prevent H-E-H-E noise
    for i in range(len(sequence)):
        start = max(0, i - 2)
        end = min(len(sequence), i + 3)
        window = sequence[start:end]
        
        avg_Pa = sum(PROPENSITIES.get(aa, {'P_a':1.0})['P_a'] for aa in window) / len(window)
        avg_Pb = sum(PROPENSITIES.get(aa, {'P_b':1.0})['P_b'] for aa in window) / len(window)
        
        if avg_Pa > 1.1 and avg_Pa > avg_Pb:
            structure.append('H') 
        elif avg_Pb > 1.1 and avg_Pb > avg_Pa:
            structure.append('E')
        else:
            structure.append('C')
    return structure

# --- 2. Geometric Templates ---

def calculate_structural_bias(moves: List[int], structure_map: List[str]) -> float:
    """
    Enforce strict geometric motifs.
    """
    energy_penalty = 0.0
    
    # Analyze windows of moves
    # Moves map to structure_map[1:] (residues 1 to N)
    
    for i in range(len(moves) - 2):
        # Context: The structure of the residue associated with move[i]
        struc_type = structure_map[i+1] # Approx alignment
        
        m1, m2, m3 = moves[i], moves[i+1], moves[i+2]
        
        if struc_type == 'H':
            # Lattice Helix Logic:
            # Constraint 1: No straight lines in helices
            if m1 == m2 or m2 == m3:
                energy_penalty += 10.0 
            
        elif struc_type == 'E':
            # Sheet Logic: Extended chain.
            # Reward consistency
            if m1 == m2:
                energy_penalty -= 5.0 # Good! Extended.
            elif m1 == m3:
                 energy_penalty -= 2.0 # Good! ZigZag (0-2-0)
            else:
                energy_penalty += 10.0 # Penalty for random turning
                
    return energy_penalty