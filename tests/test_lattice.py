import pytest
import numpy as np
import os
from protein_folding_vqe.lattice import Lattice3D

def test_lattice_initialization():
    seq = "ACD"
    lattice = Lattice3D(seq)
    assert lattice.sequence == "ACD"
    assert lattice.coordinates.shape == (3, 3)

def test_fold_sequence_valid():
    seq = "ACD" # Length 3, needs 2 moves
    lattice = Lattice3D(seq)
    # Move 0: Right (1,0,0) -> Pos 1: (1,0,0)
    # Move 2: Up (0,1,0)    -> Pos 2: (1,1,0)
    moves = [0, 2] 
    
    coords = lattice.fold_sequence(moves)
    
    assert np.array_equal(coords[0], [0, 0, 0])
    assert np.array_equal(coords[1], [1, 0, 0])
    assert np.array_equal(coords[2], [1, 1, 0])
    assert not lattice.check_self_intersection()

def test_fold_sequence_invalid_length():
    lattice = Lattice3D("ACD")
    with pytest.raises(ValueError):
        lattice.fold_sequence([0]) # Too few moves

def test_self_intersection():
    # Sequence A-B-C-D-E
    # 0,0,0 -> R -> 1,0,0 -> L -> 0,0,0 (Clash!)
    lattice = Lattice3D("ABCDE")
    moves = [0, 1, 0, 0] # Right, Left (clash), Right, Right
    lattice.fold_sequence(moves)
    assert lattice.check_self_intersection()

def test_pdb_generation(tmp_path):
    lattice = Lattice3D("AC")
    lattice.fold_sequence([0]) # 0,0,0 -> 1,0,0
    
    output_file = tmp_path / "test.pdb"
    lattice.to_pdb(str(output_file))
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "ATOM      1  CA  ALA" in content
    assert "ATOM      2  CA  CYS" in content
    assert "CONECT    1    2" in content
