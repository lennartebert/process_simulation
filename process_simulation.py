"""Helper module for the simulation of process change.
"""

import numpy as np
import networkx as nx
import pandas as pd

class ProcessSimulationPentland:
    def __init__(self, n=1000, l=100, m=1, r=50, v=0.1, max_sequence=100*10):
        """
        TODO add Docstring
        params:
            n: number of time steps
            l: lexicon: number of different process steps
            m: number of subunits of the process, needs to be a factor of l
            r: size of the history matrix
            v: chance of variation
        """
        self.n = n
        self.l = l
        self.m = m
        self.r = r
        self.v = v
        self.max_sequence = max_sequence
    
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

    def next_sequence(self, am):
        """ Perform another iteration of the simulation. Return the sequence.
        """
        global_source = 0
        global_sink = self.l - 1

        module_sinks = [int(self.l / self.m * sink) - 1 for sink in range(1, self.m+1)]
        # create a dictionary that maps each activity to a module
        activity_modules = {activity: module_sinks[int(self.m * activity / self.l)] for activity in range(0, self.l)}
        # create a dictionary that has all activities of a specific module
        module_activities = {sink_node: list(range(int(sink_node-self.l/self.m+1), sink_node+1)) for sink_node in module_sinks}
                
        current_activity = global_source
        sequence = []

        while (current_activity != global_sink) and len(sequence) < self.max_sequence-1:
            sequence.append(current_activity)

            # if the current activity is a module sink, the next activity is surely the current activity + 1
            if current_activity in module_sinks:
                next_activity = current_activity + 1
            # do not vary from usual process if a random number is > v
            elif np.random.rand() > self.v:
                # go to the next node
                # get possible next nodes from the am
                next_node_probabilities = am[current_activity,]
                
                # if the sum of the next node probabilities is 0, got to global sink
                if sum(next_node_probabilities) == 0: break
                
                next_activity = np.random.choice(list(range(self.l)), 1,
                                                 p=next_node_probabilities)[0]
            # otherwise, put in a variation within the module
            else:
                current_module = activity_modules[current_activity]
                next_activity = np.random.choice(module_activities[current_module], 1)[0]
            
            current_activity = next_activity

        sequence.append(global_sink)

        return sequence

    def run_simulation(self, record_am=[0, 100, 500, 1000]):
        # initialize the historic sequences with the happy path
        historic_sequences = [list(range(0, self.l)) for sequence in range(0, max(1, self.r))]
        
        results = {}
        for time in range(self.n + 1):
            am = self.get_adjacency_matrix(historic_sequences)
            if time in record_am:
                results[time] = am
            
            new_sequence = self.next_sequence(am)
            historic_sequences.append(new_sequence)
        
        return results
    
class ExtendedProcessSimulation:
    def __init__(self, n=1000, l=100, m=1, r=50, a=0.3, v_h=0.05, v_a=0.001, v_h_i=None, v_a_i=None, i=0, max_sequence=100*10):
        """
        TODO add Docstring
        params:
            n: number of time steps
            l: lexicon: number of different process steps
            m: number of subunits of the process, needs to be a factor of l
            r: size of the history matrix
            a: percentage of activities that are automated
            v_h: chance of variation for a human actor
            v_a: chance of variation for automated activities
            v_h_i: chance of variation for a human actor for a non-standard process input
            v_a_i: chance of variation for an automated activity for a non-standard process input
            i: chance of non-standard process input
        """
        self.n = n
        self.l = l
        self.m = m
        self.r = r
        self.a = a
        self.v_h = v_h
        self.v_a = v_a
        if v_h_i is None: v_h_i = v_h
        self.v_h_i = v_h_i
        if v_a_i is None: v_a_i = v_a
        self.v_a_i = v_a_i        
        self.i = i
        self.max_sequence = max_sequence
    
    # get the adjecency matrix for the the last r rows in the sequence history
    def get_adjacency_matrix(self, historic_sequences):
        am = np.zeros((self.l, self.l))

        # create the new adjacency matrix by adding all transitions observed in the sequence window
        # get past r observations (rows) in the historic sequences matrix
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

    def next_sequence(self, am, automated_activities):
        """ Perform another iteration of the simulation. Return the sequence.
        """
        global_source = 0
        global_sink = self.l - 1

        module_sinks = [int(self.l / self.m * sink) - 1 for sink in range(1, self.m)]
        
        is_non_standard = self.i > np.random.rand()        
        
        current_activity = global_source
        sequence = []

        while (current_activity != global_sink) and len(sequence) < self.max_sequence:
            sequence.append(current_activity)

            # if the current activity is a module sink, the next activity is surely the current activity + 1
            if current_activity in module_sinks:
                next_activity = current_activity + 1
            # do not vary from usual process if a random number is > v_h when activity is not 
            # automated or > v_a when activitiy is automated
            elif ((current_activity not in automated_activities and not is_non_standard and np.random.rand() > self.v_h) or
                  (current_activity not in automated_activities and is_non_standard and np.random.rand() > self.v_h_i) or
                  (current_activity in automated_activities and not is_non_standard and np.random.rand() > self.v_a) or
                  (current_activity in automated_activities and is_non_standard and np.random.rand() > self.v_a_i)):
                
                # go to the next node without variation
                # get possible next nodes from the am
                next_node_probabilities = am[current_activity,]
                
                # if the sum of the next node probabilities is 0, go to global sink
                if sum(next_node_probabilities) == 0: break
                
                next_activity = np.random.choice(list(range(self.l)), 1,
                                                 p=next_node_probabilities)[0]
            # otherwise, put in a variation
            else:
                next_activity = int(np.random.rand() * self.l)

            current_activity = next_activity

        sequence.append(global_sink)

        return sequence

    def run_simulation(self, record_am=[0, 100, 500, 1000]):
        node_list = list(range(0, self.l))
        
        # get the automated activities
        automated_activities = np.random.choice(node_list, int(len(node_list)*self.a))
        
        # initialize the historic sequences with the happy path
        historic_sequences = [node_list for sequence in range(0, self.r)]
        
        results = {}
        for time in range(self.n + 1):
            am = self.get_adjacency_matrix(historic_sequences)
            if time in record_am:
                results[time] = am
            
            new_sequence = self.next_sequence(am, automated_activities)
            historic_sequences.append(new_sequence)

        return results
    
    def run_experiments(self, record_am=[0, 100, 500, 1000], number_runs=30):
        experiments_results = []
        for experiment in range(number_experiments):
            sim_result = self.run_simulation(record_am)
            experiments_results.append(sim_result)
        
        # convert all to pandas dataframes
        results_dfs = []
        for experiment_results in experiments_results:
            result_summary = summarize_sim_results(experiment_results)
            results_dfs.append(result_summary)
        
        combined_dfs = pd.concat(results_dfs)
        return combined_dfs
        
    
def summarize_sim_results(adjacency_matrices):
    results = {}

    for time, adjacency_matrix in adjacency_matrices.items():
        result = {}

        # get the graph object
        graph = nx.from_numpy_matrix(adjacency_matrix)

        # calculate basic graph metrics
        number_of_nodes = graph.number_of_nodes() # this will always be l
        result['number of nodes'] = number_of_nodes
        
        number_of_connected_nodes = 0
        for node, degree in graph.degree():
            if degree > 0:
                number_of_connected_nodes += 1
        result['number of connected nodes'] = number_of_connected_nodes

        number_of_edges = graph.number_of_edges()
        result['number of edges'] = number_of_edges

        average_degree  = number_of_edges / number_of_connected_nodes
        result['average degree'] = average_degree
        
        # control flow complexity
        # consider each node with degree > 1 as a node with a choice
        # note that this is not consistent with the implementation by Mendling 2008
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
        
        # cyclomatic number
        edges_for_sequential_order = number_of_connected_nodes - 1
        cyclomatic_number = number_of_edges - edges_for_sequential_order
        
        # depth
        depth = len(nx.algorithms.shortest_paths.generic.shortest_path(graph, source=min(list(graph)), target=max(list(graph))))
        result['depth'] = depth
        
        # density
        density = number_of_edges / (number_of_connected_nodes * (number_of_connected_nodes-1))
        result['density'] = density
        
        # number s-s-paths
        # implementation by Pentland et al. 2020
        est_count_s_s_paths = 10**(0.08 + 0.08 * number_of_edges - 0.08 * number_of_connected_nodes)
        result['est. count shortest simple paths'] = est_count_s_s_paths
        
        # calculate the sum of all shortest simple paths
        # shortest_simple_paths = nx.shortest_simple_paths(graph, 0, max(graph.nodes))
        # number_shortest_simple_paths = len(list(shortest_simple_paths))
        # result['number_shortest_simple_paths'] = number_shortest_simple_paths

        results[time] = result

    # create a pandas dictionary from the results, format the dataframe here
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.transpose()
    results_df.index.rename('time', inplace=True)
    
    return results_df
