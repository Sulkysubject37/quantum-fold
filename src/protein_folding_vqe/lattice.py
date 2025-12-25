"""
3D Lattice Geometry Engine for Protein Folding.

This module handles the mapping of turn sequences to 3D Cartesian coordinates
and generates structural files (PDB) for visualization.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class Lattice3D:
    """
    Handles 3D Cubic Lattice operations for protein folding.
    
    Attributes:
        sequence (str): Amino acid sequence.
        moves (List[int]): List of direction indices defining the fold.
    """
    
    # Standard amino acid mapping (1-letter to 3-letter)
    AA_MAP = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }

    # Cubic lattice directions: (x, y, z)
    # 0: Right, 1: Left, 2: Up, 3: Down, 4: Forward, 5: Backward
    DIRECTIONS = np.array([
        [1, 0, 0],  [-1, 0, 0],
        [0, 1, 0],  [0, -1, 0],
        [0, 0, 1],  [0, 0, -1]
    ], dtype=int)

    def __init__(self, sequence: str):
        self.sequence = sequence.upper()
        self.coordinates = np.zeros((len(sequence), 3), dtype=int)
        
    def fold_sequence(self, moves: List[int]) -> np.ndarray:
        """
        Convert a sequence of moves into 3D coordinates.
        
        Args:
            moves: List of direction indices (0-5). Length must be len(sequence) - 1.
            
        Returns:
            np.ndarray: N x 3 array of integer coordinates.
            
        Raises:
            ValueError: If move count does not match sequence length.
        """
        if len(moves) != len(self.sequence) - 1:
            raise ValueError(f"Expected {len(self.sequence) - 1} moves, got {len(moves)}")
            
        # Start at origin (0,0,0)
        current_pos = np.array([0, 0, 0])
        self.coordinates[0] = current_pos
        
        for i, move_idx in enumerate(moves):
            if not (0 <= move_idx < 6):
                raise ValueError(f"Invalid move index: {move_idx}")
                
            delta = self.DIRECTIONS[move_idx]
            current_pos = current_pos + delta
            self.coordinates[i + 1] = current_pos
            
        return self.coordinates

    def check_self_intersection(self) -> bool:
        """
        Check if the folded structure intersects itself (multiple atoms on same node).
        
        Returns:
            bool: True if intersection exists (invalid structure), False otherwise.
        """
        # Unique rows in coordinates matrix
        unique_coords = np.unique(self.coordinates, axis=0)
        return len(unique_coords) != len(self.coordinates)

    def get_contact_map(self) -> List[Tuple[int, int]]:
        """
        Identify non-bonded neighbors (contacts) in the lattice.
        
        Returns:
            List of tuples (i, j) where amino acids i and j are adjacent in space
            but not adjacent in sequence (|i-j| > 1).
        """
        contacts = []
        n = len(self.sequence)
        
        for i in range(n):
            for j in range(i + 2, n): # Skip immediate neighbors
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                # In cubic lattice, neighbors have distance exactly 1.0
                if np.isclose(dist, 1.0):
                    contacts.append((i, j))
                    
        return contacts

    def to_pdb(self, filename: str = "structure.pdb") -> str:
        """
        Generate a PDB string and write to file.
        
        Args:
            filename: Output path for the PDB file.
            
        Returns:
            str: The PDB content string.
        """
        pdb_lines = []
        # Scaling factor to make visualization easier (Angstroms)
        scale = 3.0 
        
        for i, (aa, coord) in enumerate(zip(self.sequence, self.coordinates)):
            res_name = self.AA_MAP.get(aa, 'UNK')
            x, y, z = coord * scale
            
            # PDB ATOM record format
            # columns: "ATOM  ", serial, name, resName, chain, resSeq, x, y, z, occ, temp
            line = (
                f"ATOM  {i+1:5d}  CA  {res_name} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
            )
            pdb_lines.append(line)
            
        # Add CONECT records for the backbone
        for i in range(1, len(self.sequence) + 1):
            if i < len(self.sequence):
                pdb_lines.append(f"CONECT{i:5d}{i+1:5d}")
                
        content = "\n".join(pdb_lines)
        
        with open(filename, "w") as f:
            f.write(content)
            
        return content
