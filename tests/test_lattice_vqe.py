import pytest
import numpy as np
from protein_folding_vqe.vqe_optimizer import LatticeVQEOptimizer

def test_lattice_optimizer_init():
    seq = "ACD" # Length 3 -> 2 moves -> 6 qubits
    optimizer = LatticeVQEOptimizer(seq)
    assert optimizer.num_qubits == 6
    assert optimizer.ansatz.num_qubits == 6

def test_bitstring_decoding():
    optimizer = LatticeVQEOptimizer("ACD")
    # 000 010 -> Moves 0 (Right), 2 (Up)
    bits = [0, 0, 0, 0, 1, 0]
    moves = optimizer.decode_bitstring(bits)
    assert moves == [0, 2]

def test_compute_cost_run():
    # Smoke test to see if the circuit runs without crashing
    optimizer = LatticeVQEOptimizer("ACD", samples=10)
    params = optimizer.ansatz.get_initial_parameters()
    cost = optimizer.compute_cost(params)
    assert isinstance(cost, float)

def test_short_optimization():
    # Run a very short optimization to check the loop
    optimizer = LatticeVQEOptimizer("AC", samples=50) # 1 move, 3 qubits
    result = optimizer.optimize(max_iter=5)
    assert 'optimal_energy' in result
    assert 'best_moves' in result
    assert len(result['best_moves']) == 1
