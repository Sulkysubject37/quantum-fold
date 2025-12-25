import argparse
import sys
import numpy as np
from .vqe_optimizer import run_vqe_pipeline
from .folding_engines import ClassicalOptimizer, HybridOptimizer
from .interaction_models import LatticeHamiltonian
from .refinement import StructuralRefiner
from .utils import save_results

def main():
    parser = argparse.ArgumentParser(description="Protein Folding VQE (Production Grade)")
    parser.add_argument("sequence", type=str, help="Amino acid sequence (e.g., 'HHPPH')")
    parser.add_argument("--mode", type=str, choices=['lattice', 'legacy', 'classical', 'hybrid'], default='lattice', 
                       help="Simulation strategy. 'lattice'=Pure Quantum, 'classical'=Simulated Annealing, 'hybrid'=Divide & Conquer")
    
    parser.add_argument("--layers", type=int, default=1, help="Variational layers (Quantum modes)")
    parser.add_argument("--iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--chunk-size", type=int, default=7, help="Chunk size for Hybrid mode")
    
    parser.add_argument("--output", type=str, default="structure.pdb", help="Output PDB file")
    parser.add_argument("--refine", action="store_true", help="Enable Off-Lattice Refinement (High Res)")
    parser.add_argument("--json", type=str, default="results.json", help="Output JSON results")
    parser.add_argument("--verbose", action="store_true", help="Detailed logging")

    args = parser.parse_args()

    print(f"=== Quantum Protein Folding Engine ===")
    print(f"Sequence Length: {len(args.sequence)}")
    print(f"Strategy:        {args.mode.upper()}")
    if args.refine:
        print("Feature:         OFF-LATTICE REFINEMENT ENABLED")

    try:
        results = {}
        
        # Dispatch Strategy
        if args.mode == 'classical':
            optimizer = ClassicalOptimizer(args.sequence)
            results = optimizer.optimize(max_iter=args.iter)
            
        elif args.mode == 'hybrid':
            optimizer = HybridOptimizer(args.sequence, chunk_size=args.chunk_size)
            results = optimizer.optimize(vqe_layers=args.layers, vqe_iter=args.iter)
            
        else:
            # Existing Quantum Modes
            results = run_vqe_pipeline(args.sequence, num_layers=args.layers, mode=args.mode)

        print(f"\nOptimization Complete.")
        print(f"Best Energy Found: {results['optimal_energy']:.4f}")
        
        moves = results.get('best_moves')
        if moves:
            # Path 1: Refined Structure (Biological Realism)
            if args.refine:
                print("Refining structure to biological bond angles...")
                refiner = StructuralRefiner(args.sequence)
                atoms_dict = refiner.refine_structure(moves)
                refiner.to_pdb(atoms_dict, args.output)
                print(f"Refined Structure saved to: {args.output}")
                
            # Path 2: Raw Lattice Structure (Quantum Verification)
            else:
                hamiltonian = LatticeHamiltonian(args.sequence)
                try:
                    hamiltonian.lattice.fold_sequence(moves)
                    pdb_content = hamiltonian.lattice.to_pdb(args.output)
                    print(f"Lattice Structure saved to: {args.output}")
                except Exception as e:
                    print(f"Warning: Could not save PDB ({e})")
        else:
            print("No valid structure found.")

        save_results(results, args.json)

    except Exception as e:
        print(f"Critical Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()