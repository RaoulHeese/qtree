"""Tests: Quantum circuits.
"""
import pytest
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.random import random_circuit
from qtree.quant import QTreeCircuitManager

    
def test_quantum_runner_parallelization(N=10, seed=42, num_qubits=3, depth=3, backend=Aer.get_backend('aer_simulator_statevector')):
    """Equivalence of parallel and sequencial circuit evaluation.
    """
    quantum_instance = QuantumInstance(backend)
    qc_list = []
    for idx in range(N):
        qc = random_circuit(num_qubits=num_qubits, depth=depth, seed=idx+seed)
        qc_list.append(qc)  
    cm = QTreeCircuitManager()
    counts_list_par = cm.run_circuits(qc_list, quantum_instance)
    counts_list_seq = [cm.run_circuits([qc], quantum_instance)[0] for qc in qc_list]
    assert all([c1==c2 for c1, c2 in zip(counts_list_par, counts_list_seq)])    
    