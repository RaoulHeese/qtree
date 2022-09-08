"""Quantum circuit realization for quantum decision trees.
"""
import numpy as np
from itertools import product
from qiskit import QuantumCircuit
import logging


default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.WARNING)


class QTreeCircuitManager(object): 
    """Quantum circuit manager for quantum decision trees.
    
    Parameters
    ----------
    
    swap_func : callable or None
        Function (``qiskit.Circuit, q1, q2 -> None``) that defines 
        how a swap gate between two qubits with indices ``q1`` and ``q2`` 
        is realized. If set to None, 
        ``QTreeCircuitManager.gateop_swap_default`` is used. Defaults to 
        ``None``.
        
    mcswap_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list, mct_func -> None``) that 
        defines how a multi-controlled swap gate between two qubits with indices 
        ``q1`` and ``q2`` controlled by the qubits with indices ``qc_list`` is 
        realized. Also see ``mct_func`` below. If set to None,
        ``QTreeCircuitManager.gateop_mcswap_u`` is used. Defaults to ``None``.
   
    mct_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list -> None``) that defines 
        how a multi-control Toffoli gate between two qubits with indices ``q1`` 
        and ``q2`` controlled by the qubits with indices ``qc_list`` is  
        realized. If set to None, ``QTreeCircuitManager.gateop_mct_default`` 
        is used. Defaults to ``None``.
        
    logger : logging.Logger or None
        Logger for all output purposes. If ``None``, the default logger is 
        used. Defaults to ``None``.
    """

    def __init__(self, swap_func=None, mcswap_func=None, mct_func=None,
                 logger=None):
        self.swap_func = swap_func  # gateop_swap_default, gateop_swap_cnot
        self.mcswap_func = mcswap_func  # gateop_mcswap_u, gateop_mcswap_xmctx
        self.mct_func = mct_func  # gateop_mct_default
        self.logger = logger
        #
        if self.swap_func is None:
            self._swap_func = QTreeCircuitManager.gateop_swap_default
        else:
            self._swap_func = self.swap_func
        if self.mcswap_func is None:  
            self._mcswap_func = QTreeCircuitManager.gateop_mcswap_u
        else:
            self._mcswap_func = mcswap_func
        if self.mct_func is None:
            self._mct_func = QTreeCircuitManager.gateop_mct_default
        else:
            self._mct_func = mct_func
        #
        self.lg = self.logger if self.logger is not None else default_logger
        
    @staticmethod
    def gateop_swap_default(circ, q1, q2):
        """SWAP function: use default swap representation. Used as argument 
        for ``swap_func`` in ``__init__``. See there for more details. 
        """
        
        # swap gate        
        assert q1 != q2
        circ.swap(q1, q2)
        
    @staticmethod
    def gateop_swap_cnot(circ, q1, q2):  
        """SWAP function: use cnot swap representation. Used as argument 
        for ``swap_func`` in ``__init__``. See there for more details. 
        """
          
        # swap gate: cnot decomposition
        assert q1 != q2
        circ.cnot(q2, q1)
        circ.cnot(q1, q2)
        circ.cnot(q2, q1)
     
    @staticmethod
    def gateop_mcswap_u(circ, q1, q2, qc_list, mct_func):
        """MCSWAP function: use custom unitary representation. Used as argument 
        for ``mcswap_func`` in ``__init__``. See there for more details. 
        """
        
        # cswap or multi-cswap gate: unitary operator (mct_func unused)
        assert len(qc_list) > 0 and q1 != q2 and q1 not in qc_list and q2 not in qc_list
        affected_qubits = qc_list + [q1, q2]   
        n_affected_qubits = len(affected_qubits)
        n_controls = n_affected_qubits-2
        u = np.eye(2**n_affected_qubits)
        idx1 = 2**n_controls*2-1
        idx2 = 2**n_controls*3-1
        u[idx1,idx1] = 0
        u[idx2,idx2] = 0
        u[idx1,idx2] = 1
        u[idx2,idx1] = 1
        circ.unitary(u, affected_qubits)
      
    @staticmethod
    def gateop_mcswap_xmctx(circ, q1, q2, qc_list, mct_func):  
        """MCSWAP function: use cnot mcswap representation. Used as argument 
        for ``mcswap_func`` in ``__init__``. See there for more details. 
        """
        
        # cswap or multi-cswap gate: cnot + mct decomposition   
        assert len(qc_list) > 0 and q1 != q2 and q1 not in qc_list and q2 not in qc_list
        circ.cnot(q2, q1)
        mct_func(circ, q1, q2, qc_list)
        circ.cnot(q2, q1)
    
    @staticmethod
    def gateop_mct_default(circ, q1, q2, qc_list): 
        """MCT function: use default mct representation. Used as argument 
        for ``mct_func`` in ``__init__``. See there for more details. 
        """
        
        # multi-control toffoli
        assert len(qc_list) > 0 and q1 != q2 and q1 not in qc_list and q2 not in qc_list
        circ.mct(qc_list + [q1], q2) # controls, target; see: https://qiskit.org/documentation/locale/ja/api/qiskit.aqua.circuits.gates.mct.html 
    
    def _get_qubit_map(self, active_qubits_x, active_qubits_y):
        active_qubits = active_qubits_x + active_qubits_y
        qubit_map = {}
        for active_qubit in active_qubits:
            qubit_map[active_qubit] = len(qubit_map)
        return qubit_map
    
    def _find_active_qubits(self, G, Nx, Ny):
        # determine active qubits (neglecting state preparation)
        active_qubits_x = []
        for depth, g in enumerate(G):
            n_affected_qubits = depth+2
            primary_qubit = depth
            for gate_idx, secondary_qubit in enumerate(g):
                if primary_qubit == secondary_qubit: # identity
                    if primary_qubit not in active_qubits_x:
                        active_qubits_x.append(primary_qubit)
                else:
                    affected_qubits = list(range(n_affected_qubits-1)) + [secondary_qubit]
                    for qubit in affected_qubits:
                        if qubit not in active_qubits_x:
                            active_qubits_x.append(qubit)     
        active_qubits_x.sort()
        active_qubits_y = [Nx+n for n in range(Ny)]
        return active_qubits_x, active_qubits_y
    
    def _swap_data(self, X, X_swap_dict):
        if X_swap_dict is not None and len(X_swap_dict) > 0:
            X_swap = X.copy()
            for key, value in X_swap_dict.items():
                x_k = X_swap[:,key].copy()
                x_v = X_swap[:,value].copy()
                X_swap[:,value] = x_k
                X_swap[:,key] = x_v
            X = X_swap
        else:
            X = X.copy()
        return X    
    
    def _ensure_data_probabilities(self, probabilities_dict, N, c=0):
        data_probabilities = {}
        for key in product('01', repeat=N):
            data_probabilities["".join(key)] = c
        if probabilities_dict is not None:
            for key, value in probabilities_dict.items():
                key = key[::-1] # correct qiskit ordering
                data_probabilities[key] = value
        return data_probabilities

    def _to_data_probabilities(self, x):
        N = x.shape[1]   
        data_probabilities = self._ensure_data_probabilities(None, N)
        for key in product('01', repeat=N):
            key = "".join(key)
            values = [bool(int(value)) for value in key]
            p = x[np.logical_and.reduce([x[:,index]==value for index, value in enumerate(values)])].shape[0]/x.shape[0]
            key = key[::-1] # correct qiskit ordering
            data_probabilities[key] = p
        return data_probabilities   
    
    def _to_psi(self, data_probabilities, N):
        if data_probabilities is not None:
            psi = np.zeros(2**N, dtype=np.double)
            for psi_index, (key, p) in enumerate(data_probabilities.items()):
                psi[psi_index] = np.sqrt(p)
        else:
            psi = np.ones(2**N, dtype=np.double)
        psi_norm = np.sqrt(np.sum(psi*np.conj(psi)))
        psi /= psi_norm    
        return psi    
    
    def _load_data(self, X, y, active_qubits_x=None, active_qubits_y=None, X_swap_dict=None):
        Nx = X.shape[1]
        if active_qubits_x is not None:
            if len(active_qubits_x) > 0:
                X_swap = self._swap_data(X, X_swap_dict)
                X = X_swap[:,np.array(active_qubits_x)]
            else:
                X = None
        else:
            X = self._swap_data(X, X_swap_dict)
        if active_qubits_y is not None:
            if len(active_qubits_x) > 0:
                y = y[:,np.array(active_qubits_y)-Nx]
            else:
                y = None
        else:
            y = y.copy()
        if X is not None and y is not None:
            Xy = np.concatenate((X, y),axis=1)
        elif X is None and y is not None:
            Xy = y.copy()
        elif X is not None and y is None:
            Xy = X.copy()
        else:
            return None
        data_probabilities = self._to_data_probabilities(Xy)
        psi = self._to_psi(data_probabilities, Xy.shape[1])
        return psi
    
    def _apply_gate(self, circ, depth, g, qubit_map, include_id, block_form):    
        id_key = "C" # controlled swap
        nt_key = "X" # not - controlled swap - not
        n_affected_qubits = depth+2
        primary_qubit = depth
        op_code_list = ["".join(key) for key in product(nt_key+id_key, repeat=depth)] # empty defaults to swap
        if block_form:
            circ_gate = QuantumCircuit(circ.num_qubits, 0)
            active_circ = circ_gate
        else:
            active_circ = circ
        for gate_idx, secondary_qubit in enumerate(g):
            q1 = qubit_map[primary_qubit] # map
            q2 = qubit_map[secondary_qubit] # map
            op_code = op_code_list[gate_idx]
            if len(op_code) == 0: # uncontrolled swap
                qc_list = []
            else: # multi-controlled swap
                qc_list = list(range(n_affected_qubits-2))
                qc_list = [qubit_map[qubit] for qubit in qc_list] # map
            if len(op_code) > 0:
                for i, c in enumerate(op_code):
                    if c == nt_key:
                        active_circ.x(qc_list[i])
            self._mcswap(active_circ, q1, q2, qc_list, include_id)
            if len(op_code) > 0:
                for i, c in enumerate(op_code):
                    if c == nt_key:
                        active_circ.x(qc_list[i]) 
        if block_form:
            instr = active_circ.to_instruction()
            instr.name = f'<{depth}|{g}>'
            circ.append(instr, range(circ.num_qubits), [])                                
           
    def _mcswap(self, circ, q1, q2, qc_list, include_id):
        if q1 == q2: # identity
            if include_id:
                for q in qc_list + [q1]:
                    circ.i(q)
        elif len(qc_list) == 0: # uncontrolled swap
            assert len(qc_list) == 0 and q1 != q2
            self._swap_func(circ, q1, q2)  
        else: # single-/multi-controlled swap
            assert len(qc_list) > 0 and q1 != q2 and q1 not in qc_list and q2 not in qc_list
            qc_list.sort()
            self._mcswap_func(circ, q1, q2, qc_list, self._mct_func) 
               
    def build_circuit(self, G, Nx, Ny, init, measure_y=True, measure_all_x=False, remove_inactive_qubits=False, eliminate_first_swap=False, include_id=False, block_form=False, use_barriers=False):
        """Build circuit representation of decision tree.
        """
        
        assert len(G) > 0, "Empty tree cannot be built!"
        Nxy = Nx + Ny
        #
        self.lg.debug("Build circuit: Start.")
        if measure_y:
            N = Nxy
        else:
            N = Nx
        if eliminate_first_swap:
            self.lg.debug("eliminate first swap") 
            X_swap_dict = {G[0][0]: 0} # swap G[0] with 0
            G[0] = [0]
            self.lg.debug("new G = {}".format(G)) 
            self.lg.debug("X_swap_dict = {}".format(X_swap_dict)) 
        else:
            X_swap_dict = None   
        if remove_inactive_qubits:
            self.lg.debug("Build circuit: Remove inactive qubits...")
            active_qubits_x, active_qubits_y = self._find_active_qubits(G, Nx, Ny)
            if not measure_y:
                active_qubits_y = []
            if measure_all_x:
                active_qubits_x = Nx
            self.lg.debug("active_qubits_x:  {}".format(active_qubits_x))
            self.lg.debug("active_qubits_y:  {}".format(active_qubits_y))
            all_qubits = list(range(len(active_qubits_x)+len(active_qubits_y)))
            qubit_map = self._get_qubit_map(active_qubits_x, active_qubits_y)
        else:
            active_qubits_x, active_qubits_y = None, None
            all_qubits = list(range(N))
            qubit_map = {n:n for n in all_qubits}
        N_qubits = len(all_qubits)
        N_clbits = N
        
        self.lg.debug("Build circuit: Create QuantumCircuit...")
        circ = QuantumCircuit(N_qubits, N_clbits)
        if init is not None:
            if type(init) is tuple:
                X, y = init
                psi0 = self._load_data(X, y, active_qubits_x, active_qubits_y, X_swap_dict=X_swap_dict)
            elif type(init) is dict:
                self.lg.debug("convert from probability dict to array with appropriate qubit ordering")
                assert X_swap_dict is None # TODO: implement X_swap
                probabilities = init.copy()
                N_data = Nx
                probabilities = self._ensure_data_probabilities(probabilities, N_data)
                if active_qubits_x is not None:
                    self.lg.debug("reduce probabilities")
                    probabilities_reduced = {}
                    for x_str, proba in probabilities.items():
                        x_str = x_str[::-1] # remove correct qiskit ordering
                        x_reduced = [''] * len(active_qubits_x)
                        for c, idx in [(c, idx) for idx, c in enumerate(x_str) if idx in active_qubits_x]:
                            x_reduced[qubit_map[idx]] = c
                        x_str_reduced = "".join(x_reduced)
                        x_str_reduced = x_str_reduced[::-1] # restore correct qiskit ordering
                        if x_str_reduced not in probabilities_reduced:
                            probabilities_reduced[x_str_reduced] = 0
                        probabilities_reduced[x_str_reduced] += proba 
                    N_data = len(active_qubits_x)
                else:
                    probabilities_reduced = probabilities
                psi0 = self._to_psi(probabilities_reduced, N_data)
            else:
                self.lg.debug("use psi_mode as array (warning: no correction of active qubit ordering etc. is performed!)")
                psi0 = np.array(init)
            self.lg.debug("initialize with array of size {}".format(psi0.size))
            assert len(psi0.shape)==1 and psi0.size==2**len(all_qubits)
            circ.initialize(psi0, all_qubits) # TODO: use improved intialization strategy
            
        self.lg.debug("Build circuit: Build from G...")
        for depth, g in enumerate(G):
            self.lg.debug("Build circuit: Apply g #{}/{} = {}".format(depth+1, len(G), g))
            if use_barriers:
                circ.barrier(all_qubits)
            self._apply_gate(circ, depth, g, qubit_map, include_id, block_form)
        self.lg.debug("Build circuit: Measurement...")
        current_depth = len(G)-1
        if not measure_all_x:
            m_list_cl_mapped = list(range(current_depth+1))
        else:
            m_list_cl_mapped = list(range(Nx))
        if measure_y:
            m_list_cl_mapped += list(range(Nx, Nxy))
        m_list_q_mapped = [qubit_map[qubit] for qubit in m_list_cl_mapped] # map
        if use_barriers:
            circ.barrier(all_qubits)
        circ.measure(m_list_q_mapped, m_list_cl_mapped)
        
        meta = (m_list_q_mapped, m_list_cl_mapped)
        return circ, meta
        
    def run_circuits(self, circ_list, quantum_instance):
        """Evaluate a list of circuits and return the measured counts.
        """
        
        for circ in circ_list:
            circ_ops = dict(circ.count_ops())
            self.lg.debug("*\tcircuit: {}".format(circ_ops))
        result = quantum_instance.execute(circ_list)
        counts_list = []
        for circ in circ_list:
            counts = result.get_counts(circ)
            counts_list.append(counts)
        return counts_list    