# Quantum Protein Folding VQE

A production-grade framework for simulating protein folding on Quantum Computers using 3D Lattice models, Hybrid algorithms, and Biological refinement.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Cirq](https://img.shields.io/badge/quantum-cirq-green)

## Features

*   **Production Geometry Engine:**
    *   Simulates Self-Avoiding Walks (SAW) on 3D Cubic Lattices.
    *   Generates valid `.pdb` files compatible with PyMOL, Chimera, and NGLView.
*   **Physics-Based Hamiltonians:**
    *   **Hydrophobic Collapse:** Energetically favors H-H contacts in 3D space.
    *   **Secondary Structure Bias:** Uses Chou-Fasman analysis to enforce Alpha-Helix and Beta-Sheet motifs.
*   **Multi-Strategy Solvers:**
    *   **Lattice VQE (Pure Quantum):** Generative modeling using Variational Quantum Eigensolver.
    *   **Hybrid (Divide & Conquer):** Splits long proteins into quantum-solved chunks.
    *   **Classical (Simulated Annealing):** Benchmarking baseline for large proteins (50+ AA).
*   **Biological Refinement:**
    *   Post-processes lattice models into realistic off-lattice structures.
    *   Applies correct Bond Lengths ($3.8 \AA$) and Ramachandran Angles ($\phi, \psi$).

## Installation

```bash
git clone https://github.com/Sulkysubject37/protein-folding-vqe.git
cd protein-folding-vqe
pip install -e .
```

## Quick Start

### 1. Fold a small peptide (Pure Quantum)
Fold a 7-residue sequence on a local quantum simulator:
```bash
protein-vqe "HHPPHHP" --mode lattice --iter 100 --output peptide.pdb
```

### 2. Fold a medium protein (Hybrid Quantum)
Fold a 20-residue sequence using the Divide & Conquer strategy:
```bash
protein-vqe "MKTIIALSYIFCLVFADYKD" --mode hybrid --chunk-size 6 --iter 50 --output protein_hybrid.pdb
```

### 3. Fold a large protein (Classical Baseline)
Fold a 50-residue sequence using Simulated Annealing with High-Res Refinement:
```bash
protein-vqe "MKTIIALSYIFCLVFADYKDDDDKLALALALALALALALALALALALALA" --mode classical --iter 2000 --refine --output protein_real.pdb
```

## How It Works

1.  **Input:** Amino Acid Sequence (e.g., "MVLSPADKT...").
2.  **Analysis:** The `secondary_structure` module predicts propensity for Helices vs. Sheets.
3.  **Simulation:** 
    *   The **Optimizer** (Quantum or Classical) explores the landscape of 3D moves (Up, Down, Left, Right, Forward, Backward).
    *   The **Hamiltonian** calculates energy based on Hydrophobic contacts and Structural compliance.
4.  **Refinement:** The `refinement` module converts the coarse lattice path into a smooth, biological backbone.
5.  **Output:** A `.pdb` file ready for visualization.

## Documentation

Full documentation is available in the `docs/` directory. To serve locally:
```bash
mkdocs serve
```
