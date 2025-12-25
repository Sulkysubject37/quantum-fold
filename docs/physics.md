# Physics & Mathematics

## The Lattice Hamiltonian

The energy $E$ of a folded protein configuration is calculated as:

$$ E = E_{contact} + E_{bias} + E_{penalty} $$

### 1. Contact Energy ($E_{contact}$)
This term drives the hydrophobic collapse. It sums the pairwise interactions between non-bonded neighbors in the lattice.

$$ E_{contact} = \sum_{i,j \in Contacts} \epsilon_{i,j} $$

Where $\epsilon_{i,j} = -1.0 \times H(i) \times H(j)$.
$H(x)$ is the hydrophobicity of amino acid $x$.

### 2. Structural Bias ($E_{bias}$)
This term enforces secondary structure geometry. It penalizes moves that contradict the Chou-Fasman propensity of the sequence.

*   **For Helices:** Penalizes straight lines ($m_i = m_{i+1}$).
*   **For Sheets:** Penalizes turns ($m_i \neq m_{i+1}$).

$$ E_{bias} = \sum_{i} \text{Penalty}(m_i, m_{i+1} | \text{Propensity}_i) $$

### 3. Collision Penalty ($E_{penalty}$)
A massive penalty (100.0) is added if any two amino acids occupy the same $(x,y,z)$ coordinate.

## Quantum Encoding

We use a **Generative Move Encoding**.
*   A protein of length $N$ requires $N-1$ moves.
*   Each move is one of 6 directions: $+x, -x, +y, -y, +z, -z$.
*   We encode each move using **3 Qubits** ($2^3=8$ states, 6 valid).
*   Total Qubits: $3 \times (N-1)$.

The Quantum Circuit (VQE Ansatz) is trained to output bitstrings that minimize the Hamiltonian $E$.

## Refinement (The Decoder)

The lattice output gives us a topological path. The `StructuralRefiner` maps this path to continuous space:

1.  **Bond Lengths:** Fixed to $3.8 \AA$.
2.  **Torsion Angles:** 
    *   If Lattice implies a Turn $\rightarrow$ Use $(\phi, \psi)$ for Alpha Helix.
    *   If Lattice implies Extension $\rightarrow$ Use $(\phi, \psi)$ for Beta Sheet.
