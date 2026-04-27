"""
Script to calculate the Qiskit Transpiled Depth of the LR-QRC circuit.
This script extracts the core circuit logic from functional.py,
converts it to Qiskit, and transpiles it for a linear (NISQ-like) topology.
"""

import qiskit
from qiskit.transpiler import CouplingMap
from qiskit import QuantumCircuit

def build_lrqrc_circuit_qiskit(n_qubits: int) -> QuantumCircuit:
    """
    Builds one full cycle (encoding + 1 processing layer) of the LR-QRC circuit 
    directly in Qiskit to avoid any conversion issues with custom unitaries.
    """
    qc = QuantumCircuit(n_qubits)
    
    # --- 1. Encoding Layer (Fused Input + Feedback) ---
    # We use a general 2-qubit unitary to represent the fused R-gate
    # Since Qiskit will decompose it anyway, we can just use a placeholder
    # that requires decomposition, or build an equivalent depth structure.
    # To be perfectly rigorous, we apply a random 2-qubit unitary.
    from scipy.stats import unitary_group
    
    for q_i in range(n_qubits):
        # random 4x4 unitary
        U = unitary_group.rvs(4)
        target = (q_i + 1) % n_qubits
        qc.unitary(U, [q_i, target])

    # --- 2. Processing Layer (Brickwork Entanglement + Local Rotations) ---
    # Brickwork 1: Even pairs
    for j in range(0, n_qubits - 1, 2):
        qc.cx(j, j + 1)
    
    # Brickwork 2: Odd pairs
    for j in range(1, n_qubits - 1, 2):
        qc.cx(j, j + 1)
        
    # Local Rotations
    for k in range(n_qubits):
        qc.rx(0.1, k)
        qc.ry(0.1, k)
        qc.rz(0.1, k)
        
    # Brickwork 3: Odd pairs (reverse)
    for j in range(1, n_qubits - 1, 2):
        qc.cx(j, j + 1)
        
    # Brickwork 4: Even pairs (reverse)
    for j in range(0, n_qubits - 1, 2):
        qc.cx(j, j + 1)
        
    return qc

def main():
    qubit_counts = [5, 7, 8, 9, 10, 11]
    ibm_basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    
    print("="*60)
    print("LR-QRC Depth Analysis (Qiskit Native + Transpiled)")
    print("="*60)
    print(f"{'Qubits (n)':<12} | {'Logical Depth':<15} | {'Transpiled Depth (Linear Map)':<30}")
    print("-" * 60)
    
    results = {}
    for n in qubit_counts:
        # Build logical circuit
        qiskit_qc = build_lrqrc_circuit_qiskit(n)
        logical_depth = qiskit_qc.depth()
        
        # Build hardware topology (Linear map)
        linear_map = CouplingMap.from_line(n)
        
        # Transpile
        transpiled_qc = qiskit.transpile(
            qiskit_qc,
            basis_gates=ibm_basis_gates,
            coupling_map=linear_map,
            optimization_level=3,
            seed_transpiler=42 # For reproducibility
        )
        
        transpiled_depth = transpiled_qc.depth()
        results[n] = transpiled_depth
        
        print(f"{n:<12} | {logical_depth:<15} | {transpiled_depth:<30}")
        
    print("="*60)
    
    # Print dictionary format for easy copy-pasting into plot script
    print("\nCopy-paste this dictionary into your plot script if needed:")
    print("physical_depth_map = {")
    for n, depth in results.items():
        print(f"    {n}: {depth},")
    print("}")

if __name__ == "__main__":
    main()
