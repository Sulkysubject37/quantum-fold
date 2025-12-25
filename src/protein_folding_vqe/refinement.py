"""
Structure Refinement Module.
Converts coarse lattice models into All-Atom Backbone structures (N, CA, C, O).
Uses Peptidic Plane geometry to reconstruct realistic protein backbones.
"""

import numpy as np
from typing import List, Tuple, Dict
from .lattice import Lattice3D
from .secondary_structure import get_structure_propensity

class StructuralRefiner:
    """
    Refines coarse lattice models into realistic All-Atom Backbone structures.
    """
    
    # Standard Bond Lengths (Angstroms) - Engh & Huber parameters
    BOND_N_CA = 1.46
    BOND_CA_C = 1.51
    BOND_C_O  = 1.23
    BOND_C_N  = 1.33  # Peptide bond
    
    # Ideal Torsion Angles (Phi, Psi)
    ANGLES = {
        'H': (-57.0, -47.0),   # Alpha Helix (Canonical)
        'E': (-139.0, 135.0),  # Beta Sheet (Anti-parallel)
        'C': (-60.0, 120.0),   # Coil
    }

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.propensities = get_structure_propensity(sequence)

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-6: return np.array([1.0, 0.0, 0.0])
        return v / norm

    def _get_rotation_matrix(self, axis, theta):
        """Euler-Rodrigues rotation matrix."""
        axis = self._normalize(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([
            [aa+bb-cc-dd, 2*(bc-ad), 2*(bd+ac)],
            [2*(bc+ad), aa+cc-bb-dd, 2*(cd-ab)],
            [2*(bd-ac), 2*(cd+ab), aa+dd-bb-cc]
        ])

    def refine_structure(self, lattice_moves: List[int]) -> Dict[str, List[np.ndarray]]:
        """
        Reconstruct full backbone coordinates.
        Returns dictionary of atom lists: {'N': [], 'CA': [], 'C': [], 'O': []}
        """
        # Atoms storage
        atoms = {'N': [], 'CA': [], 'C': [], 'O': []}
        
        # Initial Frame (Global Origin)
        # We start with the first residue at the origin
        # N at (-1.46, 0, 0), CA at (0,0,0), C at (1.51, 0, 0)
        
        # Current Position of CA
        curr_CA = np.array([0.0, 0.0, 0.0])
        
        # Orientation vectors (NeRF frame)
        # v_bond: Vector from previous atom to current atom
        # v_plane: Normal vector to the plane of the last 3 atoms
        
        # To start, we define the N -> CA bond direction
        v_N_to_CA = np.array([1.0, 0.0, 0.0])
        v_plane = np.array([0.0, 1.0, 0.0]) # Arbitrary up
        
        # Place first N relative to CA
        curr_N = curr_CA - v_N_to_CA * self.BOND_N_CA
        
        atoms['N'].append(curr_N)
        atoms['CA'].append(curr_CA)
        
        # Previous atoms for NeRF construction
        prev_N = curr_N
        prev_CA = curr_CA
        
        for i in range(len(self.sequence)):
            # Determine geometry based on propensity and lattice
            propensity = self.propensities[i]
            phi, psi = self.ANGLES.get(propensity, self.ANGLES['C'])
            
            # Lattice Steering (if Coil)
            if propensity == 'C' and i < len(lattice_moves):
                # Simple steering: if lattice turns, we adopt a turn conformation
                move = lattice_moves[i]
                prev_move = lattice_moves[i-1] if i > 0 else move
                if move != prev_move:
                    phi, psi = (60.0, 30.0)
            
            # 1. Place C atom (defined by Psi angle)
            # We need to extend from CA.
            # Bond Angle N-CA-C is ~111 degrees (Fixed)
            # Torsion Angle is determined by previous Phi (but for first residue, arbitrary)
            
            # Construct C
            # Bond vector: CA -> C
            theta_CA = np.radians(180 - 111.0) # Deviation from straight line
            
            # Rotate v_N_to_CA around v_plane by theta_CA
            rot_bend = self._get_rotation_matrix(v_plane, theta_CA)
            v_CA_to_C = rot_bend @ v_N_to_CA
            v_CA_to_C = self._normalize(v_CA_to_C)
            
            curr_C = prev_CA + v_CA_to_C * self.BOND_CA_C
            atoms['C'].append(curr_C)
            
            # 2. Place O atom (Carbonyl Oxygen)
            # Bond length 1.23, Angle CA-C-O ~120
            # It lies roughly in the same plane as CA-C-N(next).
            # We usually place O anti-parallel to Psi rotation or simply planar.
            # Simplified: O is in the CA-C plane, bent 120 deg.
            
            v_C_to_O_dir = self._normalize(prev_CA - curr_C + np.array([0, 1, 0])) # Crude approximation
            # Better: Rotate v_CA_to_C by 120 deg in the plane
            rot_O = self._get_rotation_matrix(v_plane, np.radians(120)) # Planar
            v_C_to_O = rot_O @ (-v_CA_to_C)
            curr_O = curr_C + v_C_to_O * self.BOND_C_O
            atoms['O'].append(curr_O)
            
            # If this is the last residue, stop here
            if i == len(self.sequence) - 1:
                break
                
            # 3. Place Next N (Peptide Bond)
            # Bond C-N = 1.33
            # Angle CA-C-N ~116
            # Torsion (Omega) is usually 180 (Trans)
            
            # Rotate v_CA_to_C
            theta_C = np.radians(180 - 116.0)
            rot_peptide = self._get_rotation_matrix(v_plane, theta_C)
            v_C_to_N = rot_peptide @ v_CA_to_C
            
            # Apply Omega (Peptide planarity) - usually flat 180
            # So we keep it in the plane roughly
            
            next_N = curr_C + v_C_to_N * self.BOND_C_N
            atoms['N'].append(next_N)
            
            # 4. Place Next CA
            # Bond N-CA = 1.46
            # Angle C-N-CA ~122
            # Torsion Phi (Rotation around N-CA bond relative to C-N)
            
            # Update frame vectors
            v_prev_bond = v_C_to_N
            # Update plane normal (Cross product of last two bonds)
            v_plane = np.cross(v_CA_to_C, v_C_to_N)
            v_plane = self._normalize(v_plane)
            
            # Bending
            theta_N = np.radians(180 - 122.0)
            rot_bend_N = self._get_rotation_matrix(v_plane, theta_N)
            v_N_to_CA_new = rot_bend_N @ v_prev_bond
            
            # Torsion Phi!
            # Rotate v_N_to_CA_new around v_prev_bond (C-N) by Phi
            rot_torsion_phi = self._get_rotation_matrix(v_prev_bond, np.radians(phi))
            v_N_to_CA_new = rot_torsion_phi @ v_N_to_CA_new
            
            next_CA = next_N + v_N_to_CA_new * self.BOND_N_CA
            atoms['CA'].append(next_CA)
            
            # Torsion Psi preparation for next loop
            # Rotate the plane normal for the next step
            rot_torsion_psi = self._get_rotation_matrix(v_N_to_CA_new, np.radians(psi))
            v_plane = rot_torsion_psi @ v_plane
            
            # Update pointers
            prev_CA = next_CA
            v_N_to_CA = v_N_to_CA_new
            
        return atoms

    def to_pdb(self, atoms: Dict[str, List[np.ndarray]], filename: str) -> None:
        """Write full backbone to PDB."""
        AA_MAP = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
        
        lines = []
        
        # Header
        lines.append(f"REMARK 001 GENERATED BY QUANTUM PROTEIN FOLDING VQE")
        lines.append(f"REMARK 002 SEQUENCE: {self.sequence}")
        
        atom_serial = 1
        
        for i in range(len(self.sequence)):
            res_name = AA_MAP.get(self.sequence[i], 'UNK')
            chain = 'A'
            res_seq = i + 1
            
            # Order: N, CA, C, O
            for atom_type in ['N', 'CA', 'C', 'O']:
                if i < len(atoms[atom_type]):
                    coord = atoms[atom_type][i]
                    x, y, z = coord
                    line = (
                        f"ATOM  {atom_serial:5d}  {atom_type:<3} {res_name} {chain}{res_seq:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_type[0]}  "
                    )
                    lines.append(line)
                    atom_serial += 1
            
        # Connect
        for i in range(1, len(self.sequence)):
            # Simple connection logic: C(i-1) -> N(i)
            # Finding serial numbers...
            # Residue i-1 has 4 atoms. C is the 3rd atom (index 2). Serial = (i-1)*4 + 3
            # Residue i has N as 1st atom. Serial = i*4 + 1
            c_serial = (i-1)*4 + 3
            n_serial = i*4 + 1
            lines.append(f"CONECT{c_serial:5d}{n_serial:5d}")
            
        with open(filename, 'w') as f:
            f.write("\n".join(lines))