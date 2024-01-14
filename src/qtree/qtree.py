"""Quantum decision trees.
"""
import numpy as np
from itertools import product
import random
import copy
from deap import base, creator, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from qiskit_aer import AerSimulator
import logging
from qtree.ctree import ClassicalTree
from qtree.quant import QTreeCircuitManager

creator.create("FitnessMaxMCO", base.Fitness, weights=(1.0, 1.0, 1.0))  # positive: maximize, negative: minimize
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # positive: maximize, negative: minimize
creator.create("IndividualMCO", list, fitness=creator.FitnessMaxMCO)
creator.create("Individual", list, fitness=creator.FitnessMax)

default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.WARNING)


class QTreeBase(BaseEstimator, ClassifierMixin):
    """Base class for quantum decision classifiers.
    
    Parameters
    ----------
    
    max_depth : int
        Maximum depth of the decision tree. A tree of depth 0 consists only of 
        the root.
        
    quantum_backend : qiskit.providers.Backend
        Qiskit quantum backend used for all calculations. Can be either a
        simulator or an actual backend.
        
    d_weight : float
        Hyperparameter of tree growth. Weight of the distance function for 
        fitness calculation. Should be >= 0.
    
    s_weight : float
        Hyperparameter of tree growth. Weight of the score function for 
        fitness calculation. Should be >= 0.
    
    a_weight : float
        Hyperparameter of tree growth. Weight of the accuracy function for 
        fitness calculation. Should be >= 0.
    
    pareto_fitness : bool
        Hyperparameter of tree growth. Determines , how the three fitness 
        function components are treated. If ``False``, use scalarized fitness. 
        If ``True``, use multi-criteria fitness.
    
    score_func : callable
        Hyperparameter of tree growth. The score function maps from class 
        probabilities in a leaf to a scalar, i.e., 
        ``[py1, py2, ...] -> float``. The result measures the concentration of 
        the distribution and is used as the first fitness function component.
         
    dist_func : callable
        Hyperparameter of tree growth. The distance function maps from class 
        probabilities in two leaves to a scalar, i.e., 
        ``[py1, py2, ..., py1', py2', ..] -> float``. The result measures the 
        distance between two distributions and is used as the second fitness 
        function component.

    acc_func : callable
        Hyperparameter of tree growth. The accuracy function maps from leaf 
        to a scalar, i.e., ``[py1, py2, ...] -> float``. The result measures 
        the quality of the tree prediction and is used as the third fitness 
        function component.
        
    swap_func : callable or None
        Function (``qiskit.Circuit, q1, q2 -> None``) that defines 
        how a swap gate between two qubits with indices ``q1`` and ``q2`` 
        is realized. If set to ``None``, the callable is used, see 
        ``quant.QTreeCircuitManager``.
        
    mcswap_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list, mct_func -> None``) that 
        defines how a multi-controlled swap gate between two qubits with indices 
        ``q1`` and ``q2`` controlled by the qubits with indices ``qc_list`` is 
        realized. Also see ``mct_func`` below. If set to ``None``, the callable 
        is used, see ``quant.QTreeCircuitManager``.
   
    mct_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list -> None``) that defines 
        how a multi-control Toffoli gate between two qubits with indices ``q1`` 
        and ``q2`` controlled by the qubits with indices ``qc_list`` is  
        realized. If set to ``None``, the callable is used, see 
        ``quant.QTreeCircuitManager``.
   
    pop_size : int
        Hyperparameter of tree growth. Population size used in the evolution 
        algorithm. See ``deap`` documentation for more details.
        
    cx_pb : float
        Hyperparameter of tree growth. Crossover probability for 
        ``tools.cxOnePoint`` used in the evolution algorithm. See ``deap`` 
        documentation for more details.
        
    mut_pb : float
        Hyperparameter of tree growth. Mutation probability for 
        ``tools.mutUniformInt`` used in the evolution algorithm. See ``deap`` 
        documentation for more details.
        
    mut_ind_pb : float
        Hyperparameter of tree growth. Ìndividual gene mutation probability 
        for ``tools.mutUniformInt`` used in the evolution algorithm. See 
        ``deap`` documentation for more details.
        
    select_tournsize : int
        Hyperparameter of tree growth. Selection tournament size for 
        ``tools.selTournament`` with ``k=pop_size-1`` used in the evolution 
        algorithm. See ``deap`` documentation for more details.
        
    max_generations : int
        Hyperparameter of tree growth. Maximum number of generations used in 
        the evolution algorithm. See ``deap`` documentation for more details.

    reuse_G_calc : bool
        Hyperparameter of tree growth. Determines how tree evaluations are 
        performed. If ``True``, reuse tree evaluations. If ``False``, repeat 
        evaluations (e.g., to take noisy simulations and imperfect backends 
        into account).
    
    mean_fits : bool
        Hyperparameter of tree growth. Determines how multiple evaluation of 
        the same tree should be treated. If ``True``, fitnesses of identical 
        trees are averaged (where differences can occur due to noise). If 
        ``False``, only the newest fitness is used.
    
    random_state : int or None
        Hyperparameter of tree growth. Random seed. Can also be a 
        ``numpy.random.RandomState``. If set to ``None``, an unspecified seed 
        is used.
        
    evo_callback : callable or None
        Callback function ``(current_generation, pop) -> None`` for tree 
        growth. Called every iteration. If set to ``None``, callback function 
        is ignored.
        
    remove_inactive_qubits : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        inactive qubits are removed from the circuit. If ``False``, they are 
        included.
        
    eliminate_first_swap : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        the first swap gate is removed by changing the qubit ordering. If 
        ``False``, the first swap gate is included.
        
    simultaneous_circuit_evaluation : bool
        Determines how trees (i.e., quantum circuits) are evaluated for the 
        evolution algorithm when growing a tree. If ``True``, all circuits of 
        one iteration are evaluated simultaneously (i.e., in one job). If 
        ``False``, all circuits of one iteration are evaluated sequentially 
        (i.e., as multiple jobs).
        
    include_id : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        identity gates are included. If ``False``, they are omitted.
        
    block_form : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        tree layers are grouped and represented by custom gates. If ``False``, 
        all base gates are used instead.
        
    use_barriers : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        barriers are inserted between tree layers. If ``False``, no barriers 
        are used.
        
    logger : logging.Logger or None
        Logger for all output purposes. If ``None``, the default logger is 
        used.
    """

    def __init__(self, max_depth,
                 quantum_backend,
                 d_weight, s_weight, a_weight, pareto_fitness,
                 score_func, dist_func, acc_func, swap_func, mcswap_func, mct_func,
                 pop_size, cx_pb, mut_pb, mut_ind_pb, select_tournsize, max_generations,
                 reuse_G_calc, mean_fits,
                 random_state,
                 evo_callback,
                 remove_inactive_qubits, eliminate_first_swap, simultaneous_circuit_evaluation, include_id, block_form,
                 use_barriers,
                 logger):
        self.max_depth = max_depth
        self.quantum_backend = quantum_backend
        self.d_weight = d_weight
        self.s_weight = s_weight
        self.a_weight = a_weight
        self.pareto_fitness = pareto_fitness
        self.score_func = score_func  # e.g. entropy   
        self.dist_func = dist_func  # e.g. proba_dist_norm
        self.acc_func = acc_func  # e.g. acc_balanced   
        self.swap_func = swap_func
        self.mcswap_func = mcswap_func
        self.mct_func = mct_func
        self.pop_size = pop_size
        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.mut_ind_pb = mut_ind_pb
        self.select_tournsize = select_tournsize
        self.max_generations = max_generations
        self.reuse_G_calc = reuse_G_calc
        self.mean_fits = mean_fits
        self.random_state = random_state
        self.evo_callback = evo_callback
        self.remove_inactive_qubits = remove_inactive_qubits
        self.eliminate_first_swap = eliminate_first_swap
        self.simultaneous_circuit_evaluation = simultaneous_circuit_evaluation
        self.include_id = include_id
        self.block_form = block_form
        self.use_barriers = use_barriers
        self.logger = logger
        #
        self._score_func = lambda py: self.score_func(py)
        self._dist_func = lambda py1, py2: self.dist_func(py1, py2)
        self._acc_func = lambda y_pred, y_true: self.acc_func(y_pred, y_true)
        #
        self._cm = QTreeCircuitManager(self.swap_func, self.mcswap_func, self.mct_func, self.logger)
        self.lg = self.logger if self.logger is not None else default_logger
        #
        super().__init__()

    @staticmethod
    def check_query_probabilities(query_probabilities, Nx):
        """Check probabilistic queries (in form of dictionaries) for predictions.
        """

        assert type(query_probabilities) is dict, 'invalid query type'
        assert all([type(key) is str and len(key) == Nx and set(key) <= set('01') for key in
                    query_probabilities.keys()]), 'invalied query keys'
        assert sum(query_probabilities.values()) == 1.0, 'invalid query probabilities'
        return query_probabilities

    @staticmethod
    def check_bool_array(a):
        """Check and convert boolean array.
        """

        a = np.asarray(a).astype(bool)
        return a

    @staticmethod
    def check_input_features(X, _X):
        """Check and convert input features for predictions.
        """

        X = QTreeBase.check_bool_array(X)
        X = check_array(X)
        assert X.shape[1] == _X.shape[1], 'invalid feature dimensions'
        return X

    def _grow_ga(self, X, y, d_weight, s_weight, a_weight, pareto_fitness, max_depth, rng, pop_size, cx_pb, mut_pb,
                 mut_ind_pb, select_tournsize, max_generations, reuse_G_calc, remove_inactive_qubits,
                 eliminate_first_swap, include_id, block_form, use_barriers, simultaneous_circuit_evaluation,
                 mean_fits):
        # GA to find best decision configuration

        # Set basic options
        Nx = X.shape[1]
        Ny = y.shape[1]
        select_k = pop_size - 1
        fresh_random_seed = rng.randint(np.iinfo('int32').min, np.iinfo('int32').max)
        random.seed(fresh_random_seed)

        def G_flat_to_G(G_flat, ind_lengths):
            G_flat = G_flat.copy()
            G = []
            for length in ind_lengths:
                g = G_flat[:length]
                G_flat = G_flat[length:]
                G.append(g)
            return G

        assert select_k == pop_size - 1 and select_tournsize < pop_size - 1

        # 1) The genetic representation is a list of integers: G_flat
        ind_lengths = [2 ** (d) for d in range(max_depth + 1)]
        ind_min_values = [d for d in range(max_depth + 1)]
        ind_max_values = [Nx - 1 for _ in range(max_depth + 1)]

        # 2) The fitnes function is _calc_fitness
        G_flat_score_dict = {}  # memory: store fitness function results (to avoid recalculations)
        Q_evaluations = []

        def fitness_formater(fitness_values):
            if len(fitness_values) > 0:
                fit_str = "({})".format(",".join(["{:.4f}".format(v) for v in fitness_values]))
            else:
                fit_str = ""
            return fit_str

        def fitness_calculator(path_result_dict, G):
            path_result_dict_exact = self._get_path_result_dict_exact(path_result_dict, G, X, y)
            tree_dict, _, _ = self._to_tree_dict(G, path_result_dict)
            y_pred = self._cl_tree_prediction(tree_dict, X, y)
            y_true = y.copy()
            fitness_tuple = self._calc_fitness(path_result_dict, path_result_dict_exact, y_pred, y_true, d_weight,
                                               s_weight, a_weight)  # calc fitness
            if pareto_fitness:
                fitness_tuple = tuple(fitness_tuple)
            else:
                fitness_tuple = (float(np.sum(fitness_tuple)),)
            return fitness_tuple  # treat fitness as tuple for return value

        def fitness_function(individual):  # simultaneous_circuit_evaluation == False
            self.lg.debug("[fitness_function] Fitness calculation of one candidate:")
            G_flat = individual.copy()
            G = G_flat_to_G(G_flat, ind_lengths)
            tG_flat = tuple(G_flat)
            if tG_flat in G_flat_score_dict and reuse_G_calc:  # recall from memory
                self.lg.debug(
                    "Evaluate candidate: {} (repetition #{})".format(G, G_flat_score_dict[tG_flat]['count'] + 1))
                fitness = G_flat_score_dict[tG_flat]['fitness']
                G_flat_score_dict[tG_flat]['count'] += 1
            else:  # new, save in memory
                self.lg.debug("Run 1 quantum job (G={}):".format(G))
                path_result_dict = self._evaluate_G_list([G], Nx, Ny, init=(X, y), measure_y=True, measure_all_x=False,
                                                         remove_inactive_qubits=remove_inactive_qubits,
                                                         eliminate_first_swap=eliminate_first_swap,
                                                         include_id=include_id, block_form=block_form,
                                                         use_barriers=use_barriers)[0]
                fitness = fitness_calculator(path_result_dict, G)
                G_flat_score_dict[tG_flat] = {'fitness': fitness, 'count': 1}
                Q_evaluations.append(tG_flat)
            self.lg.debug("G={}, fitness = {}".format(G, fitness_formater(fitness)))
            return fitness  # return fitness (as tuple)

        def fitness_function_map(ind_list):  # simultaneous_circuit_evaluation == True
            self.lg.debug("[fitness_function_map] Fitness calculation: evaluate {} candidates:".format(len(ind_list)))
            self.lg.debug("Prepare evaluation...")
            fitness_job_list = []
            n_new = 0
            n_reuse = 0
            for ind_idx, individual in enumerate(ind_list):
                G_flat = individual.copy()
                G = G_flat_to_G(G_flat, ind_lengths)
                tG_flat = tuple(G_flat)
                if reuse_G_calc and tG_flat in G_flat_score_dict:
                    run_job = False
                    fetch_score_from_idx = None
                    fitness = G_flat_score_dict[tG_flat]['fitness']
                    n_reuse += 1
                elif reuse_G_calc and tG_flat in [fj['tG_flat'] for fj in fitness_job_list]:
                    run_job = False
                    fetch_score_from_idx = [fj['tG_flat'] for fj in fitness_job_list].index(tG_flat)
                    fitness = None
                    n_reuse += 1
                else:
                    run_job = True
                    fetch_score_from_idx = None
                    fitness = None
                    n_new += 1
                fitness_job = dict(G=G, G_flat=G_flat, tG_flat=tG_flat, run_job=run_job,
                                   fetch_score_from_idx=fetch_score_from_idx, fitness=fitness)
                fitness_job_list.append(fitness_job)
            self.lg.debug("Peparation complete: {} new fitnesses, {} reusable fitnesses".format(n_new, n_reuse))
            G_list = [fj['G'] for fj in fitness_job_list if fj['run_job']]
            fj_indices = [idx for idx, fj in enumerate(fitness_job_list) if fj['run_job']]
            assert len(G_list) == len(fj_indices)
            self.lg.debug("Run {} quantum jobs:".format(len(G_list)))
            path_result_dict_list = self._evaluate_G_list(G_list, Nx, Ny, init=(X, y), measure_y=True,
                                                          measure_all_x=False,
                                                          remove_inactive_qubits=remove_inactive_qubits,
                                                          eliminate_first_swap=eliminate_first_swap,
                                                          include_id=include_id, block_form=block_form,
                                                          use_barriers=use_barriers)
            self.lg.debug("Collect results...")
            fitness_list = []
            for idx, G, path_result_dict in zip(fj_indices, G_list, path_result_dict_list):
                fitness = fitness_calculator(path_result_dict, G)
                fitness_job_list[idx]['fitness'] = fitness
                tG_flat = fitness_job_list[idx]['tG_flat']
                if tG_flat not in G_flat_score_dict:
                    G_flat_score_dict[tG_flat] = {'fitness': fitness, 'count': 1}
                else:
                    G_flat_score_dict[tG_flat]['count'] += 1
                Q_evaluations.append(tG_flat)
            for idx in range(len(fitness_job_list)):
                fetch_score_from_idx = fitness_job_list[idx]['fetch_score_from_idx']
                if fetch_score_from_idx is not None:
                    fitness_job_list[idx]['fitness'] = fitness_job_list[fetch_score_from_idx]['fitness']
                fitness = fitness_job_list[idx]['fitness']
                fitness_list.append(fitness)
                self.lg.debug("fitness calc #{:d}: G={}, new={}, fitness={}".format(idx + 1, fitness_job_list[idx]['G'],
                                                                                    fitness_job_list[idx]['run_job'],
                                                                                    fitness_formater(fitness)))
            return fitness_list  # return fitnesses (as a list of tuples)

        def calculate_fitnesses(ind_list, fitness_function_map, toolbox):
            self.lg.debug("[calculate_fitnesses]")
            if simultaneous_circuit_evaluation:
                fitnesses = fitness_function_map(ind_list)
            else:
                fitnesses = list(map(toolbox.evaluate, ind_list))
            return fitnesses

        def fitness_averaging(pop):
            self.lg.debug("Fitness averaging")
            fitness_data = {}
            for ind in pop:
                G_flat = ind.copy()
                tG_flat = tuple(G_flat)
                if tG_flat not in fitness_data:
                    fitness_data[tG_flat] = []
                fitness_data[tG_flat].append(ind.fitness.values)
            self.lg.debug("fitness_data = {}".format(fitness_data))
            fitness_means, fitness_stds, fitness_counts = {}, {}, {}
            for k, v in fitness_data.items():
                fitness_means[k] = tuple(np.mean(v, axis=0).tolist())
                fitness_stds[k] = tuple(np.std(v, axis=0).tolist())
                fitness_counts[k] = len(v)
            for idx, ind in enumerate(pop):
                G_flat = ind.copy()
                tG_flat = tuple(ind)
                self.lg.debug(
                    "Candidate #{:d}: {}\n\t\t\t{} -> {} +- {}\t[x{:d}]".format(idx, G_flat_to_G(G_flat, ind_lengths),
                                                                                fitness_formater(ind.fitness.values),
                                                                                fitness_formater(
                                                                                    fitness_means[tG_flat]),
                                                                                fitness_formater(fitness_stds[tG_flat]),
                                                                                fitness_counts[tG_flat]))
                ind.fitness.values = fitness_means[tG_flat]
            return pop

        # Setup GA defined by 1) and 2)
        ind_total_length = np.sum(ind_lengths)
        ind_all_min_values = np.concatenate(
            [[min_value] * length for length, min_value in zip(ind_lengths, ind_min_values)]).tolist()
        ind_all_max_values = np.concatenate(
            [[max_value] * length for length, max_value in zip(ind_lengths, ind_max_values)]).tolist()

        def genInd(icls, ind_lengths, ind_min_values, ind_max_values):
            G_flat = []
            for length, min_value, max_value in zip(ind_lengths, ind_min_values, ind_max_values):
                g = [random.randint(min_value, max_value) for _ in range(length)]
                G_flat += g
            self.lg.debug("New candidate: {}".format(G_flat_to_G(G_flat, ind_lengths)))
            return icls(G_flat)

        toolbox = base.Toolbox()
        if pareto_fitness:
            creator_individual = creator.IndividualMCO
        else:
            creator_individual = creator.Individual
        toolbox.register('individual', genInd, creator_individual, ind_lengths, ind_min_values, ind_max_values)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", fitness_function)  # if simultaneous_circuit_evaluation == False
        toolbox.register("mate",
                         tools.cxOnePoint)  # https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.cxOnePoint
        toolbox.register("mutate", tools.mutUniformInt, low=ind_all_min_values, up=ind_all_max_values,
                         indpb=mut_ind_pb)  # https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutUniformInt
        toolbox.register("select", tools.selTournament, k=select_k,
                         tournsize=select_tournsize)  # https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament

        self.lg.info("Start of evolution")
        pop = toolbox.population(n=pop_size)

        # Evaluate the entire population
        self.lg.debug("Evaluation")
        fitnesses = calculate_fitnesses(pop, fitness_function_map, toolbox)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        self.lg.debug("Evaluated {} individuals".format(len(pop)))
        current_generation = 0
        best_fitnesses = []

        # Callback
        if self.evo_callback is not None:
            self.evo_callback(current_generation, pop)

        if pop_size > 1:

            while current_generation < max_generations:
                # A new generation
                current_generation += 1
                self.lg.debug("Generation {:2d}".format(current_generation))
                for idx, candidate in enumerate(pop):
                    self.lg.debug("Candidate #{}: {}\t{}".format(idx, G_flat_to_G(candidate, ind_lengths),
                                                                 fitness_formater(candidate.fitness.values)))

                # Select the champion from this generation
                best_champ = toolbox.clone(tools.selBest(pop, 1))
                self.lg.debug("Champion: {}\t{}".format(G_flat_to_G(best_champ[0], ind_lengths),
                                                        fitness_formater(best_champ[0].fitness.values)))

                # Select the next generation individuals
                offspring = toolbox.select(pop)
                self.lg.debug("Selection")
                for idx, candidate in enumerate(offspring):
                    self.lg.debug("Selected candidate #{}: {}\t{}".format(idx, G_flat_to_G(candidate, ind_lengths),
                                                                          fitness_formater(candidate.fitness.values)))

                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover on the offspring
                self.lg.debug("Crossover")
                crossover_idx = []
                crossover_offspring_1 = offspring[::2]
                crossover_offspring_idx_1 = list(range(len(offspring)))[::2]
                crossover_offspring_2 = offspring[1::2]
                crossover_offspring_idx_2 = list(range(len(offspring)))[1::2]
                for idx1, child1, idx2, child2 in zip(crossover_offspring_idx_1, crossover_offspring_1,
                                                      crossover_offspring_idx_2, crossover_offspring_2):
                    # cross two individuals with probability CXPB
                    if ind_total_length > 1 and random.random() <= cx_pb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                        crossover_idx.append([idx1, idx2])
                for idx, candidate in enumerate(offspring):
                    fs = fitness_formater(candidate.fitness.values)
                    if idx in [idx for idx, _ in crossover_idx]:
                        mark = "*1"
                    elif idx in [idx for _, idx in crossover_idx]:
                        mark = "*2"
                    else:
                        mark = "  "
                    self.lg.debug(
                        "Crossovered candidate #{}{}: {}\t{}".format(idx, mark, G_flat_to_G(candidate, ind_lengths),
                                                                     fs))
                self.lg.debug("Crossovers ({}: {}) vs. ({}: {}): {} ({})".format(len(crossover_offspring_idx_1),
                                                                                 crossover_offspring_idx_1,
                                                                                 len(crossover_offspring_idx_2),
                                                                                 crossover_offspring_idx_2,
                                                                                 len(crossover_idx), crossover_idx))

                # Apply mutation on the offspring
                self.lg.debug("Mutation")
                mutation_idx = []
                for idx, mutant in enumerate(offspring):
                    # mutate an individual with probability MUTPB
                    if random.random() <= mut_pb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                        mutation_idx.append(idx)
                for idx, candidate in enumerate(offspring):
                    fs = fitness_formater(candidate.fitness.values)
                    mark = "*" if idx in mutation_idx else " "
                    self.lg.debug(
                        "Mutated candidate #{}{}: {}\t{}".format(idx, mark, G_flat_to_G(candidate, ind_lengths), fs))
                self.lg.debug("Mutations: {} ({})".format(len(mutation_idx), mutation_idx))

                # Evaluate the individuals with an invalid fitness
                self.lg.debug("Evaluation")
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = calculate_fitnesses(invalid_ind, fitness_function_map, toolbox)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                self.lg.debug("Evaluated {} individuals".format(len(invalid_ind)))

                # The population is entirely replaced by the offspring and the champion of the last generation
                pop[:] = offspring + best_champ

                # Optionally agerage all fitnesses
                if mean_fits:
                    pop = fitness_averaging(pop)

                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness.values for ind in pop]
                fits_scalar = [np.sum(v) for v in fits]  # sum over all tuples
                fits_scalar_worst = np.min(fits_scalar)
                fits_scalar_best = np.max(fits_scalar)
                fits_best_idx = np.argmax(fits_scalar)
                if len(best_fitnesses) > 0:
                    if np.sum(best_fitnesses[-1]) < fits_scalar_best:
                        fit_scalar_improved = "\t(sum improved)"
                    elif np.sum(best_fitnesses[-1]) > fits_scalar_best:
                        fit_scalar_improved = "\t(sum worsened)"
                    else:
                        fit_scalar_improved = "\t(sum unchanged)"
                else:
                    fit_scalar_improved = ""  # first generation
                self.lg.debug("Fitness summary (scalarized):")
                self.lg.debug("Min: {:.5f}\t[worst]".format(fits_scalar_worst))
                self.lg.debug("Max: {:.5f}\t[best]{}".format(fits_scalar_best, fit_scalar_improved))
                self.lg.debug("Avg: {:.5f}".format(np.mean(fits_scalar)))
                self.lg.debug("Std: {:.5f}".format(np.std(fits_scalar)))
                self.lg.debug("Pop:", list(zip(pop, [fitness_formater(ind.fitness.values) for ind in pop])))
                best_fitnesses.append(pop[fits_best_idx].fitness.values)

                # Callback
                if self.evo_callback is not None:
                    self.evo_callback(current_generation, pop)
        else:
            assert pop_size == 1 and select_k == 0
            self.lg.debug(
                "Just perform a random choice since pop_size = 1 and thus select_k = 0 (i.e., selection of the empty set).")

        self.lg.info("End of evolution")

        # Summarize
        self.lg.debug("Final candidates:")
        for idx, candidate in enumerate(pop):
            self.lg.debug("Candidate #{}: {}\t{}".format(idx, G_flat_to_G(candidate, ind_lengths),
                                                         fitness_formater(candidate.fitness.values)))
        self.lg.debug("G_flat_score_dict:\n{}".format(G_flat_score_dict))
        self.lg.debug("Q_evaluations ({}):\n{}".format(len(Q_evaluations), Q_evaluations))
        self.lg.debug(
            "Progress of best fitnesses:\n{}".format(", ".join([fitness_formater(f) for f in best_fitnesses])))

        self.lg.debug("Result")

        # Select best individual
        G_flat_best = tools.selBest(pop, 1)[0]
        score_best = G_flat_best.fitness.values
        G_best = G_flat_to_G(G_flat_best, ind_lengths)
        self.lg.info("Best individual is {} with fitness {}".format(G_best, fitness_formater(score_best)))

        return G_best

    def _process_counts(self, counts, Ny, current_depth, measure_y=True):
        path_count_dict = {}
        for key, count in counts.items():
            if measure_y:
                path = tuple([int(key[-(i + 1)]) for i in range(current_depth + 1)])
                yvals = [int(key[i]) for i in range(Ny)][::-1]
                if path not in path_count_dict:
                    path_count_dict[path] = [[0, 0] for i in range(Ny)]
                for i in range(Ny):
                    path_count_dict[path][i][yvals[i]] += count  # +!
            else:
                path = tuple([int(key[-(i + 1)]) for i in range(current_depth + 1)])
                path_count_dict[path] = count
        return path_count_dict

    def _process_path_count_dict(self, path_count_dict, Ny, measure_y=True):
        path_result_dict = {}
        total_counts = np.sum([path_counts for path_counts in path_count_dict.values()])
        for path, path_counts in path_count_dict.items():
            if measure_y:
                total_path_counts = np.sum(path_counts)
                py_list = []  # one entry for all elements of y
                S_list = []  # one entry for all elements of y
                counts_list = []  # one entry for all elements of y
                for i in range(Ny):
                    counts = path_count_dict[path][i]
                    counts_list.append(counts)
                    py = np.array(counts) / np.sum(counts)  # [n_i^0(cl), n_i^1(cl)] / n_i(cl)
                    S = self._score_func(py)
                    py_list.append(py.tolist())
                    S_list.append(S)
                p = total_path_counts / total_counts  # n(cl) / N
                path_result_dict[path] = dict(p=p, py_list=py_list, S_list=S_list, N=total_counts,
                                              counts=counts_list)  # p = p(cl), py_list = [p(y_1|cl), ...], S_list=[S[p(y_1|cl)], ...]
            else:
                total_path_counts = np.sum(path_counts)
                p = total_path_counts / total_counts
                path_result_dict[path] = dict(p=p, N=total_counts)  # p = p(cl)
        return path_result_dict

    def _process_path_count_dict_list(self, path_count_dict_list, Ny, measure_y=True):
        path_result_dict_list = []
        for path_count_dict in path_count_dict_list:
            path_result_dict = self._process_path_count_dict(path_count_dict, Ny, measure_y)
            path_result_dict_list.append(path_result_dict)
        return path_result_dict_list

    def _evaluate_circuits(self, circ_list, current_depth_list, Ny, measure_y=True):
        counts_list = self._cm.run_circuits(circ_list, self.quantum_backend)
        path_count_dict_list = []
        for counts, current_depth in zip(counts_list, current_depth_list):
            path_count_dict = self._process_counts(counts, Ny, current_depth, measure_y)
            path_count_dict_list.append(path_count_dict)
        return path_count_dict_list

    def _prepare_circ_from_G(self, G, Nx, Ny, init, measure_y, measure_all_x, remove_inactive_qubits=False,
                             eliminate_first_swap=False, include_id=False, block_form=False, use_barriers=False):
        self.lg.debug("Build circuit...")
        circ, meta = self._cm.build_circuit(G, Nx, Ny, init, measure_y=measure_y, measure_all_x=measure_all_x,
                                            remove_inactive_qubits=remove_inactive_qubits,
                                            eliminate_first_swap=eliminate_first_swap, include_id=include_id,
                                            block_form=block_form, use_barriers=use_barriers)
        depth = len(G) - 1
        return circ, meta, depth

    def _evaluate_G_list(self, G_list, Nx, Ny, init, measure_y, measure_all_x, remove_inactive_qubits=False,
                         eliminate_first_swap=False, include_id=False, block_form=False, use_barriers=False):
        if len(G_list) == 0:
            self.lg.debug("Evaluate empty list of G: skip")
            path_result_dict_list = []
        else:
            self.lg.debug("Evaluate list of {} G...".format(len(G_list)))
            circ_list = []
            depth_list = []
            for idx, G in enumerate(G_list):
                self.lg.debug("Prepare {}/{}: G = {}".format(idx + 1, len(G_list), G))
                circ, _, depth = self._prepare_circ_from_G(G, Nx, Ny, init, measure_y=measure_y,
                                                           measure_all_x=measure_all_x,
                                                           remove_inactive_qubits=remove_inactive_qubits,
                                                           eliminate_first_swap=eliminate_first_swap,
                                                           include_id=include_id, block_form=block_form,
                                                           use_barriers=use_barriers)
                circ_list.append(circ)
                depth_list.append(depth)
            self.lg.debug("Evaluate circuits...")
            path_count_dict_list = self._evaluate_circuits(circ_list, depth_list, Ny, measure_y)
            self.lg.debug("Process circuit evaluation...")
            path_result_dict_list = self._process_path_count_dict_list(path_count_dict_list, Ny, measure_y)
        self.lg.debug("Finished evaluation of list of G")
        return path_result_dict_list

    def _get_Q(self, path, G, Nx):
        # get feature indices for a single path (see _G_to_features)
        assert len(G) == len(path)
        Q = list(range(Nx))
        for d, g in enumerate(G):
            if d == 0:
                index = 0
            else:
                index = 0
                for k in range(d):
                    index += path[k] * 2 ** (d - 1 - k)
            gi = g[index]
            q1 = Q[d]
            q2 = Q[gi]
            Q[d] = q2
            Q[gi] = q1
        Q = Q[:len(G)]
        return Q

    def _get_path_result_dict_exact(self, path_result_dict, G, X, y):
        def get_py_list_exact(path, G, X, y):
            Nx = X.shape[1]
            Ny = y.shape[1]
            py_list_exact = [np.nan] * Ny
            py_is_defined = [False] * Ny
            Q = self._get_Q(path, G, Nx)
            idx = np.logical_and.reduce([X[:, q] == p for q, p in zip(Q, path)])
            if len(idx) == 0:
                self.lg.debug("!")
            else:
                for yi in range(Ny):
                    count0 = np.sum(y[idx, yi] == 0)
                    count1 = np.sum(y[idx, yi] == 1)
                    total_count = (y[idx, yi]).size
                    assert count0 + count1 == total_count
                    if total_count > 0:
                        py_list_exact[yi] = [count0 / total_count, count1 / total_count]
                        py_is_defined[yi] = True
                    else:
                        py_list_exact[yi] = [.5,
                                             .5]  # undefined path (i.e. no samples with this criterion: use random guess)
                        py_is_defined[yi] = False
            return py_list_exact, py_is_defined

        # path_result_dict -> path_result_dict_exact (should lead to same result as ctree.calc_exact_probabilities)
        path_result_dict_exact = {}
        for path, value in path_result_dict.items():
            py_list_exact, py_is_defined = get_py_list_exact(path, G, X, y)
            path_result_dict_exact[path] = {'py_list': py_list_exact, 'py_is_defined': py_is_defined}
        return path_result_dict_exact

    def _py_list_list_distance(self, py_list_list1, py_list_list2):
        distance_list = []
        for py_list1, py_list2 in zip(py_list_list1, py_list_list2):
            distance_list.append(np.mean([self._dist_func(py1, py2) for py1, py2 in zip(py_list1, py_list2)]))
        return distance_list

    def _get_cl_tree(self, tree_dict, y, ct):
        if ct is None:
            ct = ClassicalTree(tree_dict, y_dim=y.shape[1], y=y, logger=self.lg)
        return ct

    def _cl_tree_prediction(self, tree_dict, X, y, ct=None, omit_warnings=True):
        ct = self._get_cl_tree(tree_dict, y, ct)
        y_pred = ct.predict(X, omit_warnings=omit_warnings)
        return y_pred

    def _cl_tree_prediction_proba(self, tree_dict, X, y, ct=None, omit_warnings=True, verbose=False):
        ct = self._get_cl_tree(tree_dict, y, ct)
        p_pred = ct.predict_proba(X, omit_warnings=omit_warnings, verbose=verbose)  # [:,:,0]
        return p_pred

    def _calc_dist_func(self, py_list_list, py_exact_list_list, p_list):
        distance_list = self._py_list_list_distance(py_list_list, py_exact_list_list)
        return np.average(distance_list, weights=p_list)

    def _calc_score_func(self, S_mean_list, p_list):
        return np.average(S_mean_list, weights=p_list)

    def _calc_acc_func(self, y_pred, y_true):
        return self._acc_func(y_pred, y_true)

    def _calc_fitness(self, path_result_dict, path_result_dict_exact, y_pred, y_true, d_weight, s_weight, a_weight):
        # remark: paths that are not sampled (i.e., not contained in path_result_dict) do not contribute to the score since they correspond to p(cl)=0 terms
        path_list = path_result_dict.keys()
        S_mean_list = np.array([np.mean(path_result_dict[path]['S_list']) for path in path_list])  # scores
        p_list = np.array([path_result_dict[path]['p'] for path in path_list]).ravel()  # weights of scores
        py_list_list = np.array([path_result_dict[path]['py_list'] for path in path_list])
        py_exact_list_list = np.array([path_result_dict_exact[path]['py_list'] for path in path_list])
        # fitness calculation (weight -1 because fitness is maximized)
        d_fitness = d_weight * -1 * self._calc_dist_func(py_list_list, py_exact_list_list, p_list)  # -> minimize
        s_fitness = s_weight * -1 * self._calc_score_func(S_mean_list, p_list)  # -> minimize
        a_fitness = a_weight * +1 * self._calc_acc_func(y_pred, y_true)  # -> maximize
        return d_fitness, s_fitness, a_fitness

    def _to_tree_dict(self, G, path_result_dict=None):
        id_key = "C"  # controlled swap
        nt_key = "X"  # not - controlled swap - not

        # genrate all paths
        max_n_qubits = np.max([np.max(g) for g in G]) + 1
        path_code_list = []
        for depth, gate_list in enumerate(G):
            op_code_list = ["".join(key) for key in product(nt_key + id_key, repeat=depth)]
            primary_qubit = depth
            path_code = [(op_code, primary_qubit, secondary_qubit) for (op_code, secondary_qubit) in
                         zip(op_code_list, gate_list)]
            path_code_list.append(path_code)

        # follow all valid paths
        tree_dict = {}
        root = tree_dict
        for path in product(*path_code_list):

            # verify path
            valid_path = True
            chosen_op_code = ""
            for path_step in path:
                (op_code, primary_qubit, secondary_qubit) = path_step
                if not np.all([c_op == op for c_op, op in zip(chosen_op_code, op_code)]):
                    valid_path = False
                    break
                chosen_op_code = op_code
            if not valid_path:
                continue

            # construct path
            qubit_order = np.arange(max_n_qubits)
            current_node = root
            for path_step_idx, path_step in enumerate(path):
                (_, primary_qubit, secondary_qubit) = path_step
                qubit_order[[primary_qubit, secondary_qubit]] = qubit_order[[secondary_qubit, primary_qubit]]
                key = qubit_order[primary_qubit]
                key = int(key)  # convert from np.int32 to int
                if key not in current_node:
                    current_node[key] = [{}, {}]  # 0, 1
                if len(chosen_op_code) > path_step_idx:
                    op = chosen_op_code[path_step_idx]
                    choice = 1 if op == id_key else 0  # op == nt_key
                    current_node = current_node[key][choice]

        # fill tree
        if path_result_dict is not None:
            tree_dict_detailed = copy.deepcopy(tree_dict)

            # tree_dict probabilities
            path_prediction_dict = {path: result['py_list'] for path, result in
                                    path_result_dict.items()}  # p(y|cl) -> tree_dict
            for path, py_list in path_prediction_dict.items():
                current_node = tree_dict
                for choice in path:
                    current_node = list(current_node.values())[0][choice]
                current_node[0] = [py[0] for py in py_list]
                current_node[1] = [py[1] for py in py_list]

                # tree_dict probabilities
            path_prediction_dict = {path: result['py_list'] for path, result in
                                    path_result_dict.items()}  # p(y|cl) -> tree_dict_detailed
            for path, py_list in path_prediction_dict.items():
                current_node = tree_dict_detailed
                for choice in path:
                    current_node = list(current_node.values())[0][choice]
                current_node['p(y|cl)'] = {0: [py[0] for py in py_list], 1: [py[1] for py in py_list]}

            # scores
            path_prediction_dict = {path: result['S_list'] for path, result in
                                    path_result_dict.items()}  # S_list -> tree_dict_detailed
            for path, S_list in path_prediction_dict.items():
                current_node = tree_dict_detailed
                for choice in path:
                    current_node = list(current_node.values())[0][choice]
                current_node['S'] = {'list': S_list, 'mean': np.mean(S_list)}

            # data probabilities
            path_prediction_dict = {path: result['p'] for path, result in
                                    path_result_dict.items()}  # p(cl) -> tree_dict_detailed
            for path, p in path_prediction_dict.items():
                current_node = tree_dict_detailed
                for choice in path:
                    current_node = list(current_node.values())[0][choice]
                current_node['p(cl)'] = p

            # counts
            path_prediction_dict = {path: result['N'] for path, result in
                                    path_result_dict.items()}  # N -> tree_dict_detailed
            for path, N in path_prediction_dict.items():
                current_node = tree_dict_detailed
                for choice in path:
                    current_node = list(current_node.values())[0][choice]
                current_node['N'] = N
        else:
            tree_dict_detailed = {}

        return tree_dict, tree_dict_detailed, path_result_dict

    def _predict_query(self, G, X, y, query_probabilities, path_result_dict=None, eliminate_first_swap=False,
                       include_id=False, block_form=False, remove_inactive_qubits=False,
                       return_additional_information=False):
        # alternative prediction method: predict query probability

        # perform measurements
        Nx = X.shape[1]
        Ny = y.shape[1]
        if path_result_dict is None:
            path_result_dict = self._evaluate_G_list([G], Nx, Ny, init=(X, y), measure_y=True, measure_all_x=False,
                                                     remove_inactive_qubits=remove_inactive_qubits,
                                                     eliminate_first_swap=eliminate_first_swap, include_id=include_id,
                                                     block_form=block_form)[0]  # optionally calculate
        path_result_dict_q = \
        self._evaluate_G_list([G], Nx, Ny, init=query_probabilities, measure_y=False, measure_all_x=False,
                              remove_inactive_qubits=remove_inactive_qubits, eliminate_first_swap=eliminate_first_swap,
                              include_id=include_id, block_form=block_form)[0]

        # accumulate results
        average_py_list = [[] for i in range(self._y.shape[1])]
        for path, path_result in path_result_dict_q.items():
            p = path_result['p']  # p^q
            if path in path_result_dict:
                py_list = path_result_dict[path]['py_list']  # p^hat
            else:
                py_list = [[.5, .5] for i in range(self._y.shape[1])]
            for i in range(self._y.shape[1]):
                average_py_list[i].append([p * py_list[i][0], p * py_list[i][1]])
        average_py_list = np.sum(average_py_list, axis=1).tolist()

        # return results
        if return_additional_information:
            additional_information = dict(path_result_dict=path_result_dict, path_result_dict_q=path_result_dict_q)
            return additional_information, average_py_list
        else:
            return average_py_list

    def _G_to_features(self, G, Nx):
        # TODO: more efficient calculation
        # convert G into feature indices (C to E in paper notation)
        C = [G[0].copy()] + [[np.nan for _ in range(2 ** k)] for k in range(1, len(G))]
        for path in product([0, 1], repeat=len(G)):
            Q = self._get_Q(path, G, Nx)
            for d in range(1, len(G)):
                index = int("".join([str(p) for p in path[:d]]), 2)
                C[d][index] = Q[d]
        return C


class QTree(QTreeBase):
    """Main class for quantum decision classifiers.
    
    Parameters
    ----------
    
    max_depth : int
        Maximum depth of the decision tree. A tree of depth 0 consists only of 
        the root.
        
    quantum_backend : qiskit.providers.Backend or None
        Qiskit quantum backend used for all calculations. Can be either a
        simulator or an actual backend. If set to ``None``, use 
        ``AerSimulator()``. Defaults to ``None``.
        
    d_weight : float
        Hyperparameter of tree growth. Weight of the distance function for 
        fitness calculation. Should be >= 0. Defaults to 1.0.
    
    s_weight : float
        Hyperparameter of tree growth. Weight of the score function for 
        fitness calculation. Should be >= 0. Defaults to 1.0.
    
    a_weight : float
        Hyperparameter of tree growth. Weight of the accuracy function for 
        fitness calculation. Should be >= 0. Defaults to 1.0.
    
    pareto_fitness : bool
        Hyperparameter of tree growth. Determines , how the three fitness 
        function components are treated. If ``False``, use scalarized fitness. 
        If ``True``, use multi-criteria fitness. Defaults to ``False``.
        
    score_func : callable or None
        Hyperparameter of tree growth. The score function maps from class 
        probabilities in a leaf to a scalar, i.e., 
        ``[py1, py2, ...] -> float``. The result measures the concentration of 
        the distribution and is used as the first fitness function component. 
        If set to ``None``, use ``QTree.score_func_entropy``. Defaults to 
        ``None``.
         
    dist_func : callable or None
        Hyperparameter of tree growth. The distance function maps from class 
        probabilities in two leaves to a scalar, i.e., 
        ``[py1, py2, ..., py1', py2', ..] -> float``. The result measures the 
        distance between two distributions and is used as the second fitness 
        function component. If set to ``None``, use 
        ``QTree.dist_func_proba_dist_norm``. Defaults to ``None``.

    acc_func : callable or None
        Hyperparameter of tree growth. The accuracy function maps from leaf 
        to a scalar, i.e., ``[py1, py2, ...] -> float``. The result measures 
        the quality of the tree prediction and is used as the third fitness 
        function component. If set to ``None``, use 
        ``QTree.acc_func_acc_balanced``. Defaults to ``None``.
        
    swap_func : callable or None
        Function (``qiskit.Circuit, q1, q2 -> None``) that defines 
        how a swap gate between two qubits with indices ``q1`` and ``q2`` 
        is realized. If set to ``None``, the callable is used, see 
        ``quant.QTreeCircuitManager``. Defaults to ``None``.
        
    mcswap_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list, mct_func -> None``) that 
        defines how a multi-controlled swap gate between two qubits with indices 
        ``q1`` and ``q2`` controlled by the qubits with indices ``qc_list`` is 
        realized. Also see ``mct_func`` below. If set to ``None``, the callable 
        is used, see ``quant.QTreeCircuitManager``. Defaults to ``None``.
   
    mct_func : callable or None
        Function (``qiskit.Circuit, q1, q2, qc_list -> None``) that defines 
        how a multi-control Toffoli gate between two qubits with indices ``q1`` 
        and ``q2`` controlled by the qubits with indices ``qc_list`` is  
        realized. If set to ``None``, the callable is used, see 
        ``quant.QTreeCircuitManager``. Defaults to ``None``.

    pop_size : int
        Hyperparameter of tree growth. Population size used in the evolution 
        algorithm. See ``deap`` documentation for more details. Defults to 10.
        
    cx_pb : float
        Hyperparameter of tree growth. Crossover probability for 
        ``tools.cxOnePoint`` used in the evolution algorithm. See ``deap`` 
        documentation for more details. Defaults to 0.2.
        
    mut_pb : float
        Hyperparameter of tree growth. Mutation probability for 
        ``tools.mutUniformInt`` used in the evolution algorithm. See ``deap`` 
        documentation for more details. Defaults to 0.2.
        
    mut_ind_pb : float
        Hyperparameter of tree growth. Ìndividual gene mutation probability 
        for ``tools.mutUniformInt`` used in the evolution algorithm. See 
        ``deap`` documentation for more details. Defaults to 0.2.
        
    select_tournsize : int
        Hyperparameter of tree growth. Selection tournament size for 
        ``tools.selTournament`` with ``k=pop_size-1`` used in the evolution 
        algorithm. See ``deap`` documentation for more details. Defaults to 3.
        
    max_generations : int
        Hyperparameter of tree growth. Maximum number of generations used in 
        the evolution algorithm. See ``deap`` documentation for more details. 
        Defaults to 10.

    reuse_G_calc : bool
        Hyperparameter of tree growth. Determines how tree evaluations are 
        performed. If ``True``, reuse tree evaluations. If ``False``, repeat 
        evaluations (e.g., to take noisy simulations and imperfect backends 
        into account). Defaults to ``True``.
    
    mean_fits : bool
        Hyperparameter of tree growth. Determines how multiple evaluation of 
        the same tree should be treated. If ``True``, fitnesses of identical 
        trees are averaged (where differences can occur due to noise). If 
        ``False``, only the newest fitness is used. Defaults to ``True``.
    
    random_state : int or None
        Hyperparameter of tree growth. Random seed. Can also be a 
        ``numpy.random.RandomState``. If set to ``None``, an unspecified seed 
        is used. Defaults to ``None``.
        
    evo_callback : callable or None
        Callback function ``(current_generation, pop) -> None`` for tree 
        growth. Called every iteration. If set to ``None``, callback function 
        is ignored. Defaults to ``None``.
        
    remove_inactive_qubits : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        inactive qubits are removed from the circuit. If ``False``, they are 
        included. Defaults to ``True``.
        
    eliminate_first_swap : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        the first swap gate is removed by changing the qubit ordering. If 
        ``False``, the first swap gate is included. Defaults to ``False``.
        
    simultaneous_circuit_evaluation : bool
        Determines how trees (i.e., quantum circuits) are evaluated for the 
        evolution algorithm when growing a tree. If ``True``, all circuits of 
        one iteration are evaluated simultaneously (i.e., in one job). If 
        ``False``, all circuits of one iteration are evaluated sequentially 
        (i.e., as multiple jobs). Defaults to ``False``.
        
    include_id : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        identity gates are included. If ``False``, they are omitted. Defaults 
        to ``False``.
        
    block_form : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        tree layers are grouped and represented by custom gates. If ``False``, 
        all base gates are used instead. Defaults to ``False``.
        
    use_barriers : bool
        Affects the circuit representation of the decision tree. If ``True``, 
        barriers are inserted between tree layers. If ``False``, no barriers 
        are used. Defaults to ``False``.
        
    logger : logging.Logger or None
        Logger for all output purposes. If ``None``, the default logger is used. 
        Defaults to ``None``.
    """

    def __init__(self, max_depth,
                 quantum_backend=None,
                 d_weight=1.0, s_weight=1.0, a_weight=1.0, pareto_fitness=False,
                 score_func=None, dist_func=None, acc_func=None, swap_func=None, mcswap_func=None, mct_func=None,
                 pop_size=10, cx_pb=.2, mut_pb=.2, mut_ind_pb=.2, select_tournsize=3, max_generations=10,
                 reuse_G_calc=True, mean_fits=True,
                 random_state=None,
                 evo_callback=None,
                 remove_inactive_qubits=True, eliminate_first_swap=False, simultaneous_circuit_evaluation=False,
                 include_id=False, block_form=False, use_barriers=False,
                 logger=None):
        #
        if quantum_backend is None:
            quantum_backend = AerSimulator()
        #
        if score_func is None:
            score_func = QTree.score_func_entropy
        if dist_func is None:
            dist_func = QTree.dist_func_proba_dist_norm
        if acc_func is None:
            acc_func = QTree.acc_func_acc_balanced
        #
        super().__init__(max_depth=max_depth,
                         quantum_backend=quantum_backend,
                         d_weight=d_weight, s_weight=s_weight, a_weight=a_weight, pareto_fitness=pareto_fitness,
                         score_func=score_func, dist_func=dist_func, acc_func=acc_func, swap_func=swap_func,
                         mcswap_func=mcswap_func, mct_func=mct_func,
                         pop_size=pop_size, cx_pb=cx_pb, mut_pb=mut_pb, mut_ind_pb=mut_ind_pb,
                         select_tournsize=select_tournsize, max_generations=max_generations,
                         reuse_G_calc=reuse_G_calc, mean_fits=mean_fits,
                         random_state=random_state,
                         evo_callback=evo_callback,
                         remove_inactive_qubits=remove_inactive_qubits, eliminate_first_swap=eliminate_first_swap,
                         simultaneous_circuit_evaluation=simultaneous_circuit_evaluation, include_id=include_id,
                         block_form=block_form, use_barriers=use_barriers,
                         logger=logger)

    @property
    def G(self):
        """list : Raw tree structure.
        """

        if hasattr(self, '_G'):
            return self._G  # g ~ c - 1 (paper notation)
        return None

    @property
    def X(self):
        """numpy.ndarray : Training data, features.
        """

        if hasattr(self, '_X'):
            return self._X
        return None

    @property
    def y(self):
        """numpy.ndarray : Training data, labels.
        """

        if hasattr(self, '_y'):
            return self._y
        return None

    @property
    def tree_dict(self):
        """dict : Tree structure as dictionary containing the class 
        probabilities of each leaf.
        """

        if hasattr(self, '_tree_dict'):
            return self._tree_dict
        return None

    @property
    def tree_dict_detailed(self):
        """dict : Tree structure as dictionary similar to ``tree_dict`` with 
        more detailed information.
        """

        if hasattr(self, '_tree_dict_detailed'):
            return self._tree_dict_detailed
        return None

    @property
    def path_result_dict(self):
        """dict : Tree traversal path summary.
        """

        if hasattr(self, '_path_result_dict'):
            return self._path_result_dict
        return None

    @property
    def ct(self):
        """ctree.ClassicalTree : Classical tree representation.
        """

        if hasattr(self, '_ct'):
            return self._ct
        return None

    @property
    def qt(self):
        """qiskit.Circuit : Circuit representation.
        """

        try:
            return self.circuit(measure_y=True, measure_all_x=False, return_meta=False, return_depth=False)
        except:
            return None

    @staticmethod
    def score_func_entropy(py):
        """Score function as argument for ``score_func`` in ``__init__``. See 
        there for more details.
        """

        # shannon information entropy (in bits)
        p0, p1 = py
        # assert p0+p1==1
        S = 0
        if p0 > 0:
            S += -p0 * np.log2(p0)
        if p1 > 0:
            S += -p1 * np.log2(p1)
        return S

    @staticmethod
    def dist_func_proba_dist_norm(py1, py2):
        """Distance function as argument for ``dist_func`` in ``__init__``.  
        See there for more details.
        """

        return np.linalg.norm(np.array(py1) - np.array(py2))

    @staticmethod
    def acc_func_acc_balanced(y_pred, y_true):
        """Accuracy function as argument for ``acc_func`` in ``__init__``.  
        See there for more details.
        """

        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        y_dim = y_pred.shape[1]
        assert y_dim == y_true.shape[1]
        return np.mean([balanced_accuracy_score(y_true[:, d], y_pred[:, d]) for d in
                        range(y_dim)])  # multi-label support: use mean

    def fit(self, X, y):
        """Fit classifier: grow the decision tree using an evolutionary 
        algorithm.
         
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Feature vectors of training data. Note: will be converted into a 
            binary array!
            
        y : array-like of shape (n_samples, n_labels)
            Target labels of training data. Note: will be converted into a 
            binary array!
            
        Returns
        -------
        
        self : returns an instance of self.
        """

        X = QTreeBase.check_bool_array(X)
        y = QTreeBase.check_bool_array(y)
        X, y = check_X_y(X, y, multi_output=True)
        self._X = X
        self._y = y
        assert self.max_depth + 1 <= self._X.shape[1], 'invalid depth for given feature vector'
        self._rng = check_random_state(self.random_state)
        self.grow(self.d_weight, self.s_weight, self.a_weight, self.pareto_fitness, self.max_depth, self._rng,
                  self.pop_size, self.cx_pb, self.mut_pb, self.mut_ind_pb, self.select_tournsize, self.max_generations,
                  self.reuse_G_calc, self.remove_inactive_qubits, self.eliminate_first_swap, self.include_id,
                  self.block_form, self.use_barriers, self.simultaneous_circuit_evaluation, self.mean_fits)
        self._tree_dict, self._tree_dict_detailed, self._path_result_dict = self.to_tree_dict(
            self.remove_inactive_qubits, self.eliminate_first_swap, self.include_id, self.block_form, self.use_barriers)
        self._ct = ClassicalTree(self._tree_dict, y_dim=self._y.shape[1], y=self._y, logger=self.lg)
        return self

    def predict(self, X, **kwargs):
        """Make class predictions using a classical representation of the 
        decision tree.
         
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Feature vectors of training data. Note: will be converted into a 
            binary array!
            
        **kwargs: 
            Additional arguments for the prediction. Will be passed to the 
            classical tree.
            
        Returns
        -------
        
        y : ndarray of shape (n_samples, n_labels)
            Predicted classes as a binary array.
        """

        check_is_fitted(self, attributes=['_X', '_y', '_tree_dict'])
        X = QTree.check_input_features(X, self._X)
        y = self.cl_tree_prediction(X, **kwargs)
        y = QTreeBase.check_bool_array(y)
        return y

    def predict_proba(self, X, **kwargs):
        """Make class probability predictions using a classical representation 
        of the decision tree.
         
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Feature vectors of training data. Note: will be converted into a 
            binary array!
            
        **kwargs: 
            Additional arguments for the prediction. Will be passed to the 
            classical tree.
            
        Returns
        -------
        
        p : ndarray of shape (n_samples, 2, n_labels)
            Predicted class probabilities.
        """

        check_is_fitted(self, attributes=['_X', '_y', '_tree_dict'])
        X = QTree.check_input_features(X, self._X)
        p = self.cl_tree_prediction_proba(X, **kwargs)
        return p

    def predict_query(self, query_probabilities):
        """Make class probability predictions using a probabilistic quantum 
        circuit query.
         
        Parameters
        ----------
        
        query_probabilities : dict
            Quantum query of the form
            ``{ key1 : probability1, key2 : probability2, ...}``, where 
            ``key`` is a bitstring of inputs/features and ``probability`` the 
            corresponding probability. The sum of all probabilities has to be 
            normalized. To predict a single key, use ``probability1 = 1.0``.
            
        Returns
        -------
        
        py : ndarray of shape (n_classes,)
            Predicted class probabilities as a binary array.
        """

        check_is_fitted(self, attributes=['_X', '_y', '_path_result_dict'])
        query_probabilities = QTree.check_query_probabilities(query_probabilities, self._X.shape[1])
        y = self.query_prediction(query_probabilities, path_result_dict=self._path_result_dict,
                                  eliminate_first_swap=self.eliminate_first_swap, include_id=self.include_id,
                                  block_form=self.block_form, remove_inactive_qubits=self.remove_inactive_qubits,
                                  return_additional_information=False)
        return np.asarray(y).ravel()

    def grow(self, d_weight, s_weight, a_weight, pareto_fitness, max_depth, rng, pop_size, cx_pb, mut_pb, mut_ind_pb,
             select_tournsize, max_generations, reuse_G_calc, remove_inactive_qubits, eliminate_first_swap, include_id,
             block_form, use_barriers, simultaneous_circuit_evaluation, mean_fits):
        """Grow decision tree. (Advanced functionality.)
        """

        self._G = self._grow_ga(self._X, self._y, d_weight, s_weight, a_weight, pareto_fitness, max_depth, rng,
                                pop_size, cx_pb, mut_pb, mut_ind_pb, select_tournsize, max_generations, reuse_G_calc,
                                remove_inactive_qubits, eliminate_first_swap, include_id, block_form, use_barriers,
                                simultaneous_circuit_evaluation, mean_fits)

    def to_tree_dict(self, remove_inactive_qubits=False, eliminate_first_swap=False, include_id=False, block_form=False,
                     use_barriers=False):
        """Build tree dict. (Advanced functionality.)
        """

        Nx = self._X.shape[1]
        Ny = self._y.shape[1]
        init = (self._X, self._y)
        path_result_dict = self._evaluate_G_list([self._G], Nx, Ny, init, measure_y=True, measure_all_x=False,
                                                 remove_inactive_qubits=remove_inactive_qubits,
                                                 eliminate_first_swap=eliminate_first_swap, include_id=include_id,
                                                 block_form=block_form, use_barriers=use_barriers)[0]
        return self._to_tree_dict(self.G, path_result_dict)

    def cl_tree_prediction(self, X, **kwargs):
        """Make class prediction with classical tree representation.
        """

        y_pred = self._cl_tree_prediction(self._tree_dict, X, self._y, ct=self._ct, **kwargs)
        return y_pred

    def cl_tree_prediction_proba(self, X, **kwargs):
        """Make class probability prediction with classical tree 
        representation.
        """

        p_pred = self._cl_tree_prediction_proba(self._tree_dict, X, self._y, ct=self._ct, **kwargs)
        return p_pred

    def query_prediction(self, query_probabilities, path_result_dict=None, eliminate_first_swap=False, include_id=False,
                         block_form=False, remove_inactive_qubits=False, return_additional_information=False):
        """Make class predictions using a probabilistic quantum circuit query. 
        """

        return self._predict_query(self._G, self._X, self._y, query_probabilities, path_result_dict,
                                   eliminate_first_swap, include_id, block_form, remove_inactive_qubits,
                                   return_additional_information)

    def G_to_features(self):
        """Extract splitting features from tree structure.
         """

        return self._G_to_features(self.G, self._X.shape[1])

    def circuit(self, measure_y=True, measure_all_x=False, return_meta=False, return_depth=False):
        """Build circuit representation of the decision tree.
         """

        Nx = self._X.shape[1]
        Ny = self._y.shape[1]
        init = (self._X, self._y)
        circ, meta, depth = self._prepare_circ_from_G(self._G, Nx, Ny, init, measure_y, measure_all_x,
                                                      remove_inactive_qubits=self.remove_inactive_qubits,
                                                      eliminate_first_swap=self.eliminate_first_swap,
                                                      include_id=self.include_id, block_form=self.block_form,
                                                      use_barriers=self.use_barriers)
        if return_meta and return_depth:
            return circ, meta, depth
        elif return_meta and not return_depth:
            return circ, meta
        elif return_depth and not return_meta:
            return circ, depth
        else:
            return circ
