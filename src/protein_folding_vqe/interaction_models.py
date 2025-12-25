"""
Amino acid interaction models for protein folding.
Includes both 1D Ising models (Legacy) and 3D Lattice models (Production).
"""

import numpy as np
import cirq
from typing import List, Tuple, Dict, Optional
from .lattice import Lattice3D
from .secondary_structure import get_structure_propensity, calculate_structural_bias

# --- Physical Constants ---
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def hydrophobicity(amino_acid: str) -> float:
    """Return hydrophobicity value for an amino acid."""
    return HYDROPHOBICITY.get(amino_acid.upper(), 0.0)

def electrostatic_potential(amino_acid: str) -> float:
    """Return electrostatic potential based on amino acid charge."""
    charge = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    return charge.get(amino_acid.upper(), 0.0)

class InteractionHamiltonian:
    """Legacy 1D Hamiltonian (for educational/testing purposes)."""
    
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.qubits = None
        
    def build_hamiltonian_terms(self, qubits):
        self.qubits = qubits
        hamiltonian_terms = []
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                aa_i = self.sequence[i % len(self.sequence)]
                aa_j = self.sequence[j % len(self.sequence)]
                hydro_weight = hydrophobicity(aa_i) * hydrophobicity(aa_j)
                zz_term = cirq.PauliString({qubits[i]: cirq.Z, qubits[j]: cirq.Z})
                hamiltonian_terms.append((hydro_weight, zz_term))
        return hamiltonian_terms

class LatticeHamiltonian:
    """
    3D Lattice Energy Model.
    Calculates energy based on spatial contacts and collision penalties.
    """

    def __init__(self, sequence: str, previous_lattice: Optional[Lattice3D] = None):
        """
        Args:
            sequence: The amino acid sequence for this segment.
            previous_lattice: A Lattice3D object containing the already folded part of the protein.
                              Used for collision detection and inter-segment contacts.
        """
        self.sequence = sequence.upper()
        self.lattice = Lattice3D(sequence)
        self.previous_lattice = previous_lattice
        
        # Pre-calculate secondary structure propensities
        self.structure_map = get_structure_propensity(self.sequence)
        
        # Hyperparameters
        self.COLLISION_PENALTY = 100.0
        self.INVALID_MOVE_PENALTY = 50.0 
        self.CONTACT_REWARD_SCALE = -1.0 

    def calculate_energy(self, moves: List[int]) -> float:
        """
        Calculate the free energy of a specific fold.
        """
        # 1. Check for invalid moves
        invalid_moves = sum(1 for m in moves if m > 5)
        if invalid_moves > 0:
            return self.INVALID_MOVE_PENALTY * invalid_moves

        # 2. Fold the current lattice segment
        try:
            current_coords = self.lattice.fold_sequence(moves)
        except ValueError:
            return self.INVALID_MOVE_PENALTY * len(moves)

        # 3. Internal Self-Intersection Check
        if self.lattice.check_self_intersection():
            return self.COLLISION_PENALTY

        # 4. Secondary Structure Bias (New!)
        bias_energy = calculate_structural_bias(moves, self.structure_map)

        # 5. Hybrid Context Check (Collision with previous segment)
        energy = 0.0 + bias_energy
        
        if self.previous_lattice is not None:
            # Shift current coordinates to start where previous lattice ended
            # The previous lattice ends at its last coordinate.
            # The current lattice starts at (0,0,0).
            # We need to translate current lattice so its origin matches previous end.
            start_offset = self.previous_lattice.coordinates[-1]
            shifted_coords = current_coords + start_offset
            
            # Use shifted coords for the actual spatial check
            
            # A. Check collisions with previous atoms
            # Broadcast check: do any shifted_coords match any previous_lattice.coordinates?
            # We skip the very first atom of shifted, because it overlaps by definition (connection point)
            # Actually, fold_sequence starts at 0,0,0. 
            # If prev lattice ends at X, the next segment starts at X.
            # So shifted_coords[0] SHOULD equal previous_lattice[-1].
            
            prev_coords = self.previous_lattice.coordinates
            
            # Check for overlaps (excluding the connection point)
            # Efficient check using set of tuples
            prev_set = set(map(tuple, prev_coords))
            curr_set = set(map(tuple, shifted_coords[1:])) # Skip start node
            
            if not prev_set.isdisjoint(curr_set):
                return self.COLLISION_PENALTY

            # B. Calculate Inter-segment Contacts
            # Check neighbors between shifted_coords and prev_coords
            for i, c_pos in enumerate(shifted_coords[1:]): # Skip start
                for j, p_pos in enumerate(prev_coords[:-1]): # Skip connection
                    dist = np.linalg.norm(c_pos - p_pos)
                    if np.isclose(dist, 1.0):
                        # Found a contact!
                        aa_i = self.sequence[i+1] # +1 because we skipped start
                        aa_j = self.previous_lattice.sequence[j]
                        h_i = hydrophobicity(aa_i)
                        h_j = hydrophobicity(aa_j)
                        if (h_i * h_j) > 0:
                             energy += self.CONTACT_REWARD_SCALE * (h_i * h_j)

        # 5. Internal Contact Energy
        contacts = self.lattice.get_contact_map()
        for i, j in contacts:
            h_i = hydrophobicity(self.sequence[i])
            h_j = hydrophobicity(self.sequence[j])
            if (h_i * h_j) > 0:
                energy += self.CONTACT_REWARD_SCALE * (h_i * h_j)

        return energy