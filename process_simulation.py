"""Helper module for the simulation of process change.
"""

import numpy as np
import networkx as nx
import pandas as pd
import random
import h5py
import itertools
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


VERSION = 0.02

def sample_adjacency_matrices(ams, sample_size):
    # retrieves "sample_size" items from ams
    n = len(ams)
    if sample_size > n:
        raise ValueError("sample_size cannot be greater than the number of entries in ams")

    # Ensure first and last entries are included
    indices = [0]  # Start with the first index

    # Calculate the step size
    if sample_size > 2:
        step_size = (n - 1) / (sample_size - 1)
        for i in range(1, sample_size - 1):
            indices.append(round(i * step_size))
    
    indices.append(n-1)  # End with the last index
    
    # Remove duplicates if any (this can happen if rounding causes duplicates)
    indices = sorted(set(indices))

    # Retrieve the corresponding entries from the dictionary
    samples = {index: ams[index] for index in indices}

    return samples

class ProcessSimulationModel:
    def __init__(self, t, l, m, r, n, v_m, v_a=0, v_m_e=0.01, v_a_e=0.03, a=0, e=0, seed=None):
        """
        TODO add Docstring
        params:
            t: number of time steps
            l: lexicon: number of different process steps
            m: number of subunits of the process, needs to be a factor of l
            r: size of the history matrix
            v: chance of variation
            n: maximal sequence length factor (max. sequence length is n * l)
            seed: random seed for reproducibility
        """
        self.t = t
        self.l = l
        self.m = m
        self.r = r
        self.n = n
        self.a = a
        self.e = e
        self.v_m = v_m
        self.v_a = v_a
        self.v_m_e = v_m_e
        self.v_a_e = v_a_e
        self.seed = seed
    
    # get the adjecency matrix for the the last r rows in the sequence history
    def get_adjacency_matrix(self, historic_sequences):
        am = np.zeros((self.l, self.l))

        # create the new adjacency matrix by adding all transitions observed in the sequence window
        # get past r observations (rows) in the historic sequences matrix
        # regard the special case where r=0 - no historic sequences have an impact
        window_sequences = None
        if self.r == 0:
            window_sequences = [historic_sequences[0]]
        else:
            window_sequences = historic_sequences[-self.r:]
        
        for sequence in window_sequences:
            for sequence_position, activity in enumerate(sequence):
                if sequence_position + 1 >= len(sequence): break  # stop if the last activity is reached
                am[activity, sequence[sequence_position + 1]] += 1

        # normalize the adjacency matrix by rows
        sum_of_rows = am.sum(axis=1)
        sum_of_rows_matrix = sum_of_rows[:, np.newaxis]
        norm_am = np.divide(am, sum_of_rows_matrix, out=np.zeros_like(am), where=sum_of_rows_matrix != 0)

        return norm_am

    def get_v(self, activitiy, is_activity_automated, is_exception):
        if (is_activity_automated[activitiy]) and (is_exception): return self.v_a_e
        elif (is_activity_automated[activitiy]) and (not is_exception): return self.v_a
        elif (not is_activity_automated[activitiy]) and (is_exception): return self.v_m_e
        elif (not is_activity_automated[activitiy]) and (not is_exception): return self.v_m

    def next_sequence(self, am, global_source, global_sink, module_sinks, activity_modules, module_activities, is_activity_automated):
        """ Perform another iteration of the simulation. Return the sequence.
        """
        current_activity = global_source
        sequence = [current_activity]

        is_exception = self.e > np.random.rand()
        max_sequence_length = self.n * self.l

        while (current_activity != global_sink) and (len(sequence) < max_sequence_length):
            # if the current activity is a module sink, the next activity is surely the current activity + 1
            if current_activity in module_sinks:
                next_activity = current_activity + 1
            # do not vary from usual process if a random number is > variability
            elif np.random.rand() > self.get_v(current_activity, is_activity_automated, is_exception):
                # go to the next node
                # get possible next nodes from the am
                next_node_occurences = am[current_activity,]
                sum_next_node_occurences = np.sum(next_node_occurences)
                
                # if the sum of the next node occurences is 0, perform a random jump in the module (in accordance with Pentlands implementation)
                if sum_next_node_occurences == 0:
                    current_module = activity_modules[current_activity]
                    next_activity = random.choice(module_activities[current_module])
                # else, select the next node per weighted jump
                else:
                    # custom implementation of random.choice to pick next activity based on non-normalized values
                    cumulative_occurences = np.cumsum(next_node_occurences)
                    random_value = np.random.rand() * sum_next_node_occurences
                    next_activity = np.searchsorted(cumulative_occurences, random_value)

            # otherwise, put in a variation within the module
            else:
                current_module = activity_modules[current_activity]
                next_activity = random.choice(module_activities[current_module])

            current_activity = next_activity
            sequence.append(current_activity)

        return sequence
    
    def run_simulation(self, normalize_adjacency_matrices=True):
        # set the seed at each simulation run for reproducability
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # simulate each time step
        historic_adjacency_matrices = []
        # historic_sequences_incl_init = [list(range(self.l))*(_+1) for _ in range(self.r)] # code for testing a unique sequence for initialization 
        historic_sequences_incl_init = [list(range(self.l)) for _ in range(self.r)]

        # initialize global variables
        global_source = 0
        global_sink = self.l - 1

        module_sinks = [int(self.l / self.m * sink) - 1 for sink in range(1, self.m+1)]
        # create a dictionary that maps each activity to a module
        activity_modules = {activity: module_sinks[int(self.m * activity / self.l)] for activity in range(0, self.l)}
        # create a dictionary that has all activities of a specific module
        module_activities = {sink_node: list(range(int(sink_node-self.l/self.m+1), sink_node+1)) for sink_node in module_sinks}
        
        # create dictionary that holds which activities are automated
        is_activity_automated = np.random.rand(self.l) < self.a

        for i in range(self.t):
            # build adjacency matrix
            am = None

            # special case if r=0 - no variation
            if self.r == 0:
                # adjacency matrix is always the same happy path
                if i == 0:
                    am = np.zeros((self.l, self.l))
                    for activity in range(self.l-1):
                        am[activity][activity+1] = 1
                else: am = np.copy(historic_adjacency_matrices[-1])
            elif i == 0:
                # initialize the adjacency matrix with the happy path
                am = np.zeros((self.l, self.l))
                # for activity in range(self.l-1):
                #     am[activity][activity+1] = self.r
                for sequence in historic_sequences_incl_init:
                    for index, activity_from in enumerate(sequence[:-1]):
                        activity_to = sequence[index+1]
                        am[activity_from][activity_to] += 1
            else:
                # get the last adjacency matrix
                am = np.copy(historic_adjacency_matrices[-1])

                # forget sequence that is now out of the retention window
                sequence_to_forget = historic_sequences_incl_init[-self.r-1]
                # remove the sequence
                for index, activity_from in enumerate(sequence_to_forget[:-1]):
                    activity_to = sequence_to_forget[index+1]
                    am[activity_from][activity_to] -= 1
                
                # add last sequence to matrix
                last_sequence = historic_sequences_incl_init[-1]
                for index, activity_from in enumerate(last_sequence[:-1]):
                    activity_to = last_sequence[index+1]
                    am[activity_from][activity_to] += 1

            # get next sequence based on adjacency matrix
            next_sequence = self.next_sequence(am, global_source, global_sink, module_sinks, activity_modules, module_activities, is_activity_automated)

            # write the historic data
            historic_sequences_incl_init.append(next_sequence)
            historic_adjacency_matrices.append(am)

            # integrity check: are only past r sequences in history matrix? identify by occurence counts
            # optional code for debugging
            # last_am = historic_adjacency_matrices[-1]
            # total_occurences_in_am = np.sum(last_am)
            # relevant_historic_sequences = historic_sequences_incl_init[-self.r-1:-1]
            # total_occurences_in_last_r_iters = sum([len(sequence)-1 for sequence in relevant_historic_sequences])
            # if (total_occurences_in_last_r_iters != total_occurences_in_am): raise Exception("AM not in line with past occurences")
        
        sim_result = np.array(historic_adjacency_matrices)

        if normalize_adjacency_matrices:
            # calculate the nomarlized adjacency matrixes for report out
            norm_adjacency_matrices = []
            for am in historic_adjacency_matrices:
                # normalize the adjacency matrix by rows
                sum_of_rows = am.sum(axis=1)
                sum_of_rows_matrix = sum_of_rows[:, np.newaxis]
                norm_am = np.divide(am, sum_of_rows_matrix, out=np.zeros_like(am), where=sum_of_rows_matrix != 0)
                norm_adjacency_matrices.append(norm_am)
            sim_result = norm_adjacency_matrices
        
        return sim_result
    
def get_number_of_connected_nodes(graph):
    number_of_connected_nodes = 0
    for node, degree in graph.degree():
        if degree > 0:
            number_of_connected_nodes += 1
    return number_of_connected_nodes

def summarize_am(adjacency_matrix, metrics=['number of connected nodes', 'number of edges', 'density', 'avg degree', 'est. count simple paths'], monte_carlo_max_sequence=None, monte_carlo_iterations = 1000):
    result = {}

    # get the graph object
    graph = nx.from_numpy_array(adjacency_matrix)

    # always get number of nodes and edges
    number_of_nodes = graph.number_of_nodes()
    number_of_edges = graph.number_of_edges()

    # calculate basic graph metrics
    if 'number of nodes' in metrics:
        result['number of nodes'] = number_of_nodes
    
    if 'number of connected nodes' in metrics:
        result['number of connected nodes'] = get_number_of_connected_nodes(graph)

    if 'number of edges' in metrics:
        result['number of edges'] = number_of_edges

    if 'avg degree' in metrics:
        number_of_connected_nodes = None
        if 'number_of_connected_nodes' in result:
            number_of_connected_nodes = result['number_of_connected_nodes']
        else:
            number_of_connected_nodes = get_number_of_connected_nodes(graph)

        average_degree  = number_of_edges / number_of_connected_nodes
        result['avg degree'] = average_degree
    
    # control flow complexity
    # consider each node with degree > 1 as a node with a choice
    # note that this is not consistent with the implementation by Mendling 2008
    if 'control flow complexity' in metrics:
        choices = 0
        for node, degree in graph.degree():
            if degree > 1:
                choices += 1
        control_flow_complexity = choices
        result['control flow complexity'] = control_flow_complexity
    
    # cyclicity
    # Ratio of activities on a cycle
    # use networkx.algorithms.cycles.find_cycle to find a cycle starting at node_i
    # whenever a cycle is found, we know that all other nodes on the cycle are also on a cycle
    if 'cyclicity' in metrics:
        number_of_connected_nodes = None
        if 'number_of_connected_nodes' in result:
            number_of_connected_nodes = result['number_of_connected_nodes']
        else:
            number_of_connected_nodes = get_number_of_connected_nodes(graph)

        number_nodes_on_cycle = 0
        node_cycle_candidates = list(graph)
        nodes_on_cycle = set()
        
        while len(node_cycle_candidates) > 0:
            node_cycle_candidate = node_cycle_candidates[0]
            # try/catch because networkx.algorithms.cycles.find_cycle() posts an error if no cycle is found 
            try:
                cycle = nx.algorithms.cycles.find_cycle(graph, node_cycle_candidate)
                for node_on_cycle in cycle:
                    node_cycle_candidates.remove(node_on_cycle)
                
                nodes_on_cycle.update(cycle)
            except:
                node_cycle_candidates.pop()
                
        cyclicity = len(nodes_on_cycle) / number_of_connected_nodes
        result['cyclicity'] = cyclicity

    
    # depth
    if 'depth' in metrics:
        depth = len(nx.algorithms.shortest_paths.generic.shortest_path(graph, source=min(list(graph)), target=max(list(graph))))
        result['depth'] = depth
    
    # density
    if 'density' in metrics:
        number_of_connected_nodes = None
        if 'number_of_connected_nodes' in result:
            number_of_connected_nodes = result['number_of_connected_nodes']
        else:
            number_of_connected_nodes = get_number_of_connected_nodes(graph)
        
        density = number_of_edges / (number_of_connected_nodes * (number_of_connected_nodes-1))
        result['density'] = density
    
    # number of shortest simple paths
    # implementation by Pentland et al. 2020
    
    if 'est. count simple paths' in metrics:
        est_count_s_paths = 10**(0.08 + 0.08 * number_of_edges - 0.08 * number_of_nodes) 
        result['est. count simple paths'] = est_count_s_paths

    if 'count simple paths' in metrics:
        all_simple_paths = list(nx.all_simple_paths(graph, source=0, target=number_of_nodes-1))
        count_simple_paths = len(all_simple_paths)
        result['count simple paths'] = count_simple_paths
        
    # calculate probabilistic metrics by performing monte carlo runs through the adjacency matrix and then count the averages

    # perform monte_carlo_count monte carlo runs through the adjacency matrix (source to sink with cut-off)
    
    if ('avg sequences ending in sink' in metrics) or ('avg sequences with loops' in metrics) or ('avg steps to sink' in metrics):
        monte_carlo_sequences = []

        for _ in range(monte_carlo_iterations):
            monte_carlo_sequences.append(get_random_walk_sequence(adjacency_matrix, max_sequence))
        
        # Average number of sequences ending in sink
        if 'avg sequences ending in sink' in metrics:
            sequence_count_ending_in_sink = 0
            for sequence in monte_carlo_sequences:
                if (sequence[len(sequence) - 1]) == (len(adjacency_matrix) - 1): sequence_count_ending_in_sink += 1

            result['avg sequences ending in sink'] = sequence_count_ending_in_sink / monte_carlo_iterations

        # average number of sequences with loops (defined as recurring activities in sequences)
        if 'avg sequences with loops' in metrics:
            sequence_count_loop = 0
            for sequence in monte_carlo_sequences:
                if len(sequence) != len(set(sequence)): sequence_count_loop += 1

            result['avg sequences with loops'] = sequence_count_loop / monte_carlo_iterations

        # average number of steps to sink
        if 'avg steps to sink' in metrics:
            steps_to_sink = 0
            for sequence in monte_carlo_sequences:
                steps_to_sink += len(sequence)

            result['avg steps to sink'] = steps_to_sink / monte_carlo_iterations
        
    return result

def get_aggregate_sim_result(adjacency_matrices, l):
    complexities = []
    max_complexity = None

    # estimate complexity based on edge counts
    for adjacency_matrix in adjacency_matrices:
        edges = np.count_nonzero(adjacency_matrix)
        est_count_s_paths = 10**(0.08 + 0.08 * edges - 0.08 * l) 
        complexity = est_count_s_paths
        if (max_complexity is None) or (complexity > max_complexity):
            max_complexity = complexity
        complexities.append(complexity)
    
    # evaluate whether there has been a phase change

    # criteria 1: a peak of 3 orders of magnitude of the average complexity
    # criteria 2: did the complexity after the last peak reduce below the level before the first peak
    
    has_phase_change = False

    # Step 1: Calculate the mean of the array
    mean_complexity = np.mean(complexities)

    # Step 2: Find the indices where the value is greater than 3 times the mean
    threshold = 3 * mean_complexity
    indices = np.where(complexities > threshold)[0]

    criteria_1 = len(indices) > 0

    if criteria_1:
        mean_complexity_before_first_peak = np.mean(complexities[:indices[0]])
        mean_complexity_after_last_peak = np.mean(complexities[indices[-1]:])
        criteria_2 = mean_complexity_after_last_peak < mean_complexity_before_first_peak
        if criteria_2:
            has_phase_change = True

    # Determine time to chaos, the number of iterations until process complexity reaches the threshold for the first time
    time_to_chaos = indices[0] if len(indices) > 0 else None

    return {'max_complexity': max_complexity, 'mean_complexity': mean_complexity, 'has_phase_change': has_phase_change, 'time_to_chaos': time_to_chaos}

def get_metrics_for_sim_results(adjacency_matrices, metrics=['number of connected nodes', 'number of edges', 'density', 'avg degree', 'est. count simple paths'], monte_carlo_max_sequence=None, monte_carlo_iterations = 1000):
    results = []

    for adjacency_matrix in adjacency_matrices:
        am_summary = summarize_am(adjacency_matrix, metrics, monte_carlo_max_sequence, monte_carlo_iterations)
        results.append(am_summary)

    # create a pandas dictionary from the results, format the dataframe here
    results_df = pd.DataFrame(results)
    # results_df = results_df.transpose()
    results_df.index.rename('time', inplace=True)
    
    return results_df

def get_random_walk_sequence(am, max_sequence=None):
        """ Perform a random walk through the given adjacency matrix, always starting at global source
        """
        # derive l from adjacency matrix
        num_rows, num_cols = am.shape
        if num_rows != num_cols: raise Exception("Adjacency matrix not symmetrical")
        l = num_rows

        # set max_sequence if None
        if max_sequence is None: max_sequence = 5 * num_rows

        global_source = 0
        
        # set current activity to global source
        current_activity = global_source
        
        # initialize sequence with global source
        sequence = [current_activity]

        # break criteria for while loop: no next nodes
        no_next_node = False

        while len(sequence) < max_sequence and (not no_next_node):
            # go to the next node

            # get possible next nodes from the am
            next_node_probabilities = am[current_activity,]
            
            # if the sum of the next node probabilities is 0, set no next node to break the while loop
            if sum(next_node_probabilities) == 0: 
                no_next_node=True
            else:
                next_activity = np.random.choice(list(range(l)), 1,
                                                    p=next_node_probabilities)[0]
                
                current_activity = next_activity
                sequence.append(current_activity)

        return sequence

def save_simulation_hdf5_to_open_file(f, model, results, computation_time):
    """
    Saves a simulation run to an HDF5 file with an automatically assigned unique ID.
    - Model parameters are stored as attributes.
    - Simulation results are stored in a separate subgroup to avoid confusion.

    :param f: Open HDF5 file.
    :param model: ProcessSimulationModel instance.
    :param results: Dictionary containing summarized simulation results.
    :return: Assigned simulation ID.
    """
    # Generate a unique ID
    simulation_id = f"simulation_{len(f.keys()) + 1}"

    # Create a group for this simulation
    group = f.create_group(simulation_id)

    # Store model parameters as attributes
    for key, value in vars(model).items():
        group.attrs[key] = value

    # store the version of the simulation
    group.attrs['version'] = VERSION

    # Create a subgroup for results
    results_group = group.create_group("results")

    # Store results in the subgroup
    for key, value in results.items():
        results_group.attrs[key] = value

    # Create a subgroup for results
    compute_info_group = group.create_group("compute_info")
    compute_info_group.attrs['computation_time'] = computation_time
    compute_info_group.attrs['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    return simulation_id  # Return the generated ID

def save_simulation_hdf5(filename, model, results, computation_time):
    """
    Saves a simulation run to an HDF5 file with an automatically assigned unique ID.
    - Model parameters are stored as attributes.
    - Simulation results are stored in a separate subgroup to avoid confusion.

    :param filename: HDF5 file path.
    :param model: ProcessSimulationModel instance.
    :param results: Dictionary containing summarized simulation results.
    :return: Assigned simulation ID.
    """
    with h5py.File(filename, "a") as f:
        simulation_id = save_simulation_hdf5_to_open_file(f, model, results, computation_time)

    return simulation_id  # Return the generated ID

def load_simulation_hdf5(filename, simulation_id):
    """
    Loads a specific simulation run from an HDF5 file.

    :param filename: HDF5 file path.
    :param simulation_id: Unique ID of the simulation.
    :return: ProcessSimulationModel instance, NumPy activity matrices, and results dictionary.
    """
    with h5py.File(filename, "r") as f:
        if simulation_id not in f:
            raise ValueError(f"Simulation ID {simulation_id} not found.")

        group = f[simulation_id]

        # Load model parameters and reconstruct the model
        model_kwargs = {key: group.attrs[key] for key in group.attrs}

        # remove version from model_kwargs
        model_kwargs.pop('version')

        model = ProcessSimulationModel(**model_kwargs)

        # Load summarized results
        results_group = group["results"]
        results = {}
        for key in results_group.attrs:
            results[key] = results_group.attrs[key]  # Load simple attributes
        for key in results_group:
            results[key] = results_group[key][()]  # Load NumPy datasets

        # Load summarized results
        compute_info_group = group["compute_info"]
        compute_info = {}
        for key in compute_info_group.attrs:
            compute_info[key] = compute_info_group.attrs[key]  # Load simple attributes
        for key in compute_info_group:
            compute_info[key] = compute_info_group[key][()]  # Load NumPy datasets
   
    return model, results, compute_info

def load_simulations_by_params(filename, **search_params):
    """
    Loads all simulations that match the given parameters.

    :param filename: HDF5 file path.
    :param search_params: Key-value pairs of parameters to filter by.
    :return: List of tuples (simulation_id, ProcessSimulationModel instance, results)
    """
    matching_simulations = []

    with h5py.File(filename, "r") as f:
        for sim_id in f.keys():
            group = f[sim_id]

            # Extract stored parameters
            model_kwargs = {key: group.attrs[key] for key in group.attrs}
            
            # remove version from model_kwargs
            model_kwargs.pop('version')

            # Check if this simulation matches the given search parameters
            if all(model_kwargs.get(key) == value for key, value in search_params.items()):

                model = ProcessSimulationModel(**model_kwargs)
                
                # Load summarized results
                results_group = group["results"]
                results = {}
                for key in results_group.attrs:
                    results[key] = results_group.attrs[key]
                for key in results_group:
                    results[key] = results_group[key][()]

                matching_simulations.append((sim_id, model, results))

    return matching_simulations

def create_results_dataframe(filename):
    """
    Creates a DataFrame where:
    - The index consists of all model parameters.
    - The columns contain all results

    :param filename: HDF5 file path.
    :return: Pandas DataFrame.
    """
    records = []
    model_params = None

    with h5py.File(filename, "r") as f:
        for sim_id in f.keys():
            group = f[sim_id]

            # Extract model parameters
            model_params = {key: group.attrs[key] for key in group.attrs}
            model_params['sim_id'] = sim_id

            # Extract summarized results
            results_group = group["results"]
            results = {}
            for key in results_group.attrs:
                results[key] = results_group.attrs[key]  # Scalars
            for key in results_group:
                results[key] = results_group[key][()].tolist()  # Convert arrays to lists

            # Combine parameters and results
            record = {**model_params, **results}
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Set model parameters as index
    param_keys = list(model_params.keys())
    df.set_index(param_keys, inplace=True)

    return df

def run_simulation_and_get_results(sim_params):
    # Time the simulation
    start_time = time.time()

    # Create simulation instance
    my_simulation = ProcessSimulationModel(**sim_params)
    # Run simulation
    adjacency_matrices = my_simulation.run_simulation()

    # Get aggregate simulation result
    simulation_result = get_aggregate_sim_result(adjacency_matrices, my_simulation.l)
    
    end_time = time.time()

    computation_time = end_time - start_time
    return my_simulation, adjacency_matrices, simulation_result, computation_time

def run_simulations_to_hdf5(simulation_settings, filename="results.h5"):
    """
    Runs simulations in parallel and writes each result immediately to an HDF5 file.
    This prevents accumulating all results in memory.
    """
    
    # Generate parameter sets

    simulation_runs = simulation_settings['simulation_runs']  # Number of times each combination should appear
    params = simulation_settings['params']

    params_list = [
        dict(zip(params.keys(), values))
        for values in itertools.product(*params.values())
        for _ in range(simulation_runs)
    ]

    # Total number of simulations, used for progress tracking
    total_simulations = len(params_list)
    total_simulations_str = str(total_simulations)


    print("Starting simulation")
    start_time = time.time()

    with h5py.File(filename, "w") as h5file:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(run_simulation_and_get_results, params): params for params in params_list}
            for future in as_completed(futures):
                my_simulation, adjacency_matrices, simulation_result, computation_time = future.result()

                # Save the simulation results to the HDF5 file
                save_simulation_hdf5_to_open_file(h5file, my_simulation, adjacency_matrices, simulation_result, computation_time)

                # Update progress bar
                elapsed_time = time.time() - start_time
                simulations_performed = len(h5file.keys())
                percent_progress = simulations_performed / total_simulations

                remaining_time = elapsed_time / percent_progress - elapsed_time
                remaining_time_struct = time.gmtime(remaining_time)
                remaining_time_str = time.strftime("%H:%M:%S", remaining_time_struct)
                simulations_performed_str = str(simulations_performed).zfill(len(total_simulations_str))
                if percent_progress < 1:
                    print(f"Progress: {percent_progress:2.1%} | Simulations: {simulations_performed_str}/{total_simulations_str} | Estimated remaining time: {remaining_time_str}", end="\r")
                else:
                    # clear line
                    print(" " * 200, end="\r")
                    print(f"Progress: {percent_progress:2.1%} | Simulations: {simulations_performed_str}/{total_simulations_str} | Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

def main(simulation_settings, filename="results.h5"):
    """
    Main function that accepts the simulation settings and runs the simulations.
    """

    run_simulations_to_hdf5(simulation_settings, filename)
    print(f"All simulations complete and results saved to {filename}")

if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Run simulations in parallel and save results to an HDF5 file."
    )
    parser.add_argument(
        "--settings", type=str, required=True,
        help="Path to a YAML file containing the simulation settings."
    )
    parser.add_argument(
        "--output", type=str, default="results.h5",
        help="Output HDF5 file name (default: results.h5)."
    )
    args = parser.parse_args()

    # Load parameter sets from the provided JSON file.
    with open(args.settings, "r") as f:
        simulation_settings = yaml.safe_load(f)

    main(simulation_settings, args.output)
