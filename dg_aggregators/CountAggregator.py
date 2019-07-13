from dg_aggregators.AbstractAggregator import AbstractAggregator
from collections import defaultdict


class CountAggregator(AbstractAggregator):

    """
    An implementation of the count aggregator class
    """

    def __init__(self, dep_graphs, cls):
        """
        Keep count of every relevant node seen in all input dependency graphs
        :param dep_graphs: input dependency graphs
        :param cls: the class these dependency graphs correspond to
        """
        self.cls = cls
        self.node_counts = defaultdict(int)
        self.n_graphs = 0

        for dep_graph in dep_graphs:
            self.update(dep_graph)


    def update(self, dep_graph):
        """
        Update neuron counts
        :param dep_graph: new dependency graph
        """

        self.n_graphs += 1

        for (l, relevance) in dep_graph.items():
            for node in relevance:
                self.node_counts[(l, node)] += 1



    def similarity(self, dep_graph):
        """
        Compute similarity score between previously-seen dependency graphs
        and new dependency graph, based on how frequently nodes in the new
        dependency graph were seen before

        :param dep_graph: new dependency graph
        :return: similarity score between dep_graph and previously-seen graphs
        """


        # TODO: consider increasing the similarity weights of neurons closer to the output layer
        sim_score = 0
        node_count = 0

        # Sum up similarity scores for all nodes
        for (l, relevance) in dep_graph.items():
            for node in relevance:
                sim_score += self.node_counts[(l, node)]
                node_count += 1

        return sim_score
