from dg_aggregators.AbstractAggregator import AbstractAggregator



class ProbAggregator(AbstractAggregator):

    """
    An implementation of the abstract aggregator class
    """

    def __init__(self, dep_graphs):
        # TODO: create probabilistic node sets
        pass

    def update(self, dep_graph):
        # TODO: update probabilistic graph
        pass

    def similarity(self, dep_graph):
        # TODO: compute similarity of input graph with stored probabilistic graph
        pass
