

class AbstractAggregator():

    """
    Abstract class specifying methods necessary for dependency graph aggregators
    """

    @abstractmethod
    def __init__(self, dep_graphs):
        """
        Aggregate a set of dependency graphs
        :param dep_graphs: set of dependency graphs to process
        """
        pass


    @abstractmethod
    def update(self, dep_graph):
        """
        Update the stored representation with a new dependency graph
        :param dep_graph: new dependency graph
        """
        pass

    @abstractmethod
    def similarity(self, dep_graph):
        """
        Compute the similarity of an input dependency graph with the stored
        representation of previously seen dependency graphs
        :param dep_graph: input dependency graph
        :return:
        """
        pass
