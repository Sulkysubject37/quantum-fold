# Welcome to Protein Folding VQE

This project bridges the gap between **Quantum Computing** and **Structural Biology**. It provides a complete pipeline for predicting protein structures using lattice-based models that are amenable to quantum simulation.

## Core Philosophy

Real protein folding is driven by:
1.  **Hydrophobic Collapse:** The protein folds to hide hydrophobic residues from water.
2.  **Hydrogen Bonding:** Local interactions form regular structures like Helices and Sheets.
3.  **Steric Constraints:** Atoms cannot overlap.

This software models all three using a **3D Cubic Lattice Hamiltonian**:
*   **Hydrophobicity** is modeled by the HP (Hydrophobic-Polar) interaction matrix.
*   **Secondary Structure** is modeled by geometric biases derived from sequence propensity.
*   **Steric Constraints** are enforced by infinite energy penalties for self-intersection.

## Supported Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Lattice** | Full Quantum VQE simulation. | Small peptides (<8 AA), Research. |
| **Hybrid** | Divide & Conquer VQE. Splits protein into chunks. | Medium proteins (10-50 AA). |
| **Classical** | Simulated Annealing. | Large proteins (50+ AA), Benchmarking. |

## Next Steps

Check out the [User Guide](user-guide.md) to get started.