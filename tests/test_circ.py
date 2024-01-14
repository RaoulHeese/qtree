"""Tests: Quantum circuits.
"""
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qtree.quant import QTreeCircuitManager


def test_quantum_circuits_swap(N=2, q1=0, q2=1):
    """Equivalence of SWAP representations.
    """
    qc1 = QuantumCircuit(N)
    qc2 = QuantumCircuit(N)
    QTreeCircuitManager.gateop_swap_default(qc1, q1, q2)
    QTreeCircuitManager.gateop_swap_cnot(qc2, q1, q2)
    assert Statevector.from_instruction(qc1).equiv(Statevector.from_instruction(qc2))


def test_quantum_circuits_mct(N=4, q1=0, q2=1, qc_list=[2, 3]):
    """Equivalence of MCT representations.
    """
    qc1 = QuantumCircuit(N)
    qc2 = QuantumCircuit(N)
    mct_func = QTreeCircuitManager.gateop_mct_default
    QTreeCircuitManager.gateop_mcswap_u(qc1, q1, q2, qc_list, mct_func)
    QTreeCircuitManager.gateop_mcswap_xmctx(qc2, q1, q2, qc_list, mct_func)
    assert Statevector.from_instruction(qc1).equiv(Statevector.from_instruction(qc2))
