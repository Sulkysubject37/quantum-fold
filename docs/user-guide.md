# User Guide

The primary interface is the `protein-vqe` Command Line Interface (CLI).

## Basic Usage

```bash
protein-vqe [SEQUENCE] [OPTIONS]
```

## Arguments

*   `SEQUENCE`: The amino acid sequence string (e.g., "MKTIIALSY"). Case-insensitive.

## Options

### Simulation Modes
*   `--mode lattice` (Default): Uses the `LatticeVQEOptimizer`. This runs a full Variational Quantum Eigensolver on the simulator.
*   `--mode hybrid`: Uses the `HybridOptimizer`. Breaks the sequence into overlapping chunks and solves each with VQE.
*   `--mode classical`: Uses the `ClassicalOptimizer` (Simulated Annealing). Fast and robust for large proteins.

### Optimization Parameters
*   `--iter INT`: Maximum number of optimization iterations. Default: 100.
*   `--layers INT`: Number of variational layers (depth of quantum circuit). Default: 1.
*   `--chunk-size INT`: Size of fragments for Hybrid mode. Default: 7.

### Output Control
*   `--output FILE`: Path to save the generated PDB file. Default: `structure.pdb`.
*   `--json FILE`: Path to save numerical results. Default: `results.json`.
*   `--refine`: **Crucial Flag.** Enables the Off-Lattice Refinement engine. If set, the output PDB will have realistic biological geometry. If unset, it will be a blocky lattice model.
*   `--verbose`: Prints detailed error traces and progress logs.

## Examples

**1. High-Resolution Simulation of a Beta Sheet:**
```bash
protein-vqe "VIVIVIVIVI" --mode classical --iter 2000 --refine --output sheet.pdb
```

**2. Quantum Simulation of a Helix:**
```bash
protein-vqe "AAAAAAAA" --mode lattice --iter 100 --output helix_lattice.pdb
```
