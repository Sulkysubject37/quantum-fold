"""
Folding Engines for Protein Structure Prediction.
Includes Classical (Simulated Annealing) and Hybrid (Divide & Conquer) strategies.
"""

import numpy as np
import random
import copy
from typing import List, Dict, Optional, Any
from .interaction_models import LatticeHamiltonian
from .lattice import Lattice3D
from .vqe_optimizer import LatticeVQEOptimizer

class ClassicalOptimizer:
    """
    Simulated Annealing Optimizer for Protein Folding.
    Good for long sequences where Quantum simulation is infeasible.
    """
    
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.hamiltonian = LatticeHamiltonian(sequence)
        self.num_moves = len(sequence) - 1
        
    def optimize(self, max_iter: int = 1000, temp_start: float = 10.0, temp_end: float = 0.1) -> Dict[str, Any]:
        """Run Simulated Annealing."""
        print(f"Running Classical Simulated Annealing for {len(self.sequence)} residues...")
        
        # 1. Initialize random valid moves
        current_moves = [random.randint(0, 5) for _ in range(self.num_moves)]
        current_energy = self.hamiltonian.calculate_energy(current_moves)
        
        best_moves = list(current_moves)
        best_energy = current_energy
        
        history = [current_energy]
        
        # Cooling schedule
        cooling_rate = (temp_end / temp_start) ** (1 / max_iter)
        temp = temp_start
        
        for i in range(max_iter):
            # 2. Perturb: Flip one move
            new_moves = list(current_moves)
            idx_to_change = random.randint(0, self.num_moves - 1)
            new_moves[idx_to_change] = random.randint(0, 5)
            
            # 3. Calculate Energy
            new_energy = self.hamiltonian.calculate_energy(new_moves)
            
            # 4. Metropolis Criterion
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < np.exp(-delta_E / temp):
                current_moves = new_moves
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_moves = list(current_moves)
            
            history.append(current_energy)
            temp *= cooling_rate
            
        return {
            'optimal_energy': best_energy,
            'best_moves': best_moves,
            'history': history,
            'success': True,
            'message': 'Simulated Annealing Completed'
        }

class HybridOptimizer:
    """
    Divide and Conquer Strategy.
    Breaks a long sequence into chunks, solves them with Quantum VQE, 
    and stitches them together.
    """
    
    def __init__(self, sequence: str, chunk_size: int = 7):
        self.sequence = sequence
        self.chunk_size = chunk_size
        
    def optimize(self, vqe_layers: int = 1, vqe_iter: int = 50) -> Dict[str, Any]:
        print(f"Running Hybrid Quantum-Classical Folding for {len(self.sequence)} residues.")
        print(f"Chunk size: {self.chunk_size} residues.")
        
        global_moves = []
        global_lattice = Lattice3D(self.sequence[0:1]) # Start with just the first residue
        # Actually, we don't need to prepopulate.
        
        # Split sequence into chunks
        # Chunk 1: 0..K
        # Chunk 2: K..2K, etc.
        # But they must overlap by 1 residue (the connection point)
        
        # Example: Size 5. 
        # Seq: ABCDEFGHI
        # C1: ABCDE (0-4)
        # C2: EFGHI (4-8) - E is shared anchor
        
        start_idx = 0
        current_lattice_context = None # Accumulates the full structure
        
        while start_idx < len(self.sequence) - 1:
            end_idx = min(start_idx + self.chunk_size, len(self.sequence))
            
            # Identify sub-sequence
            sub_seq = self.sequence[start_idx : end_idx]
            
            # We skip the first char of sub_seq if it's not the very first chunk
            # because it's just the anchor.
            # Wait, the LatticeVQEOptimizer expects a sequence and produces moves for it.
            # If we pass "ABCDE", it gives moves for A->B, B->C...
            # If we pass "EFGHI", it gives moves for E->F...
            # This is exactly what we want.
            
            print(f"Folding Chunk: {sub_seq} ({start_idx}-{end_idx})")
            
            # Instantiate VQE for this chunk
            # We must modify VQE to accept a 'previous_lattice' context if we want robust collision checking
            # But LatticeVQEOptimizer uses LatticeHamiltonian.
            # We can patch the Hamiltonian inside the optimizer!
            
            vqe = LatticeVQEOptimizer(sub_seq, num_layers=vqe_layers)
            
            # Inject context
            if current_lattice_context:
                vqe.hamiltonian.previous_lattice = current_lattice_context
                
            # Run VQE
            result = vqe.optimize(max_iter=vqe_iter)
            
            chunk_moves = result.get('best_moves')
            
            # If VQE fails to find a valid move (rare but possible), fallback to simple straight line?
            if not chunk_moves:
                 print("Warning: Quantum Chunk failed. Defaulting to straight line.")
                 chunk_moves = [0] * (len(sub_seq) - 1)
            
            # Append moves
            global_moves.extend(chunk_moves)
            
            # Update Context
            # We need to rebuild the FULL lattice up to this point to be the context for the next
            # Sub-lattice is not enough.
            # So we create a Lattice for 0..end_idx
            
            full_sub_seq = self.sequence[0:end_idx]
            current_lattice_context = Lattice3D(full_sub_seq)
            current_lattice_context.fold_sequence(global_moves)
            
            # Move start index
            # Overlap is 1 residue. So if we did 0-5 (Length 5), next is 4-9.
            # end_idx is exclusive. 
            start_idx = end_idx - 1
            
            # Safety break for infinite loops
            if len(sub_seq) < 2: 
                break
                
        # Final calculation
        final_ham = LatticeHamiltonian(self.sequence)
        final_energy = final_ham.calculate_energy(global_moves)
        
        return {
            'optimal_energy': final_energy,
            'best_moves': global_moves,
            'success': True,
            'message': 'Hybrid Folding Completed'
        }
