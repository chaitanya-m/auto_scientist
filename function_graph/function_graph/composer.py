from abc import ABC, abstractmethod
from typing import List, Dict, overload
from function_graph.node import FunctionNode

class GraphFunctionComposer(ABC):
    """
    Interface for composing and compressing function graphs.
    """

    @overload
    def add_function(self, node: FunctionNode) -> FunctionNode:
        ...  # For the initial node (empty graph)

    @overload
    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode]) -> FunctionNode:
        ...  # For subsequent nodes (non-empty graph)

    @abstractmethod
    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode] = None) -> FunctionNode:
        """
        Adds a function node to the graph.

        This method is overloaded to handle two cases:
        1. Adding the first node to an empty graph: input_nodes should be None.
        2. Adding subsequent nodes: input_nodes must be provided.

        Args:
            node: The FunctionNode to add.
            input_nodes: List of FunctionNodes that provide inputs to this node.
                         Required for non-empty graphs.

        Returns:
            The added FunctionNode.

        Raises:
            ValueError: If input_nodes is not provided for a non-empty graph.
        """
        pass

    @abstractmethod
    def get_function(self, name: str) -> FunctionNode:
        """
        Retrieves a function node from the graph by its name.

        Args:
            name: The name of the node to retrieve.

        Returns:
            The FunctionNode with the given name.
        """
        pass

    @abstractmethod
    def compress_graph(self, nodes: List[FunctionNode], name: str) -> FunctionNode:
        """
        Compresses a subgraph into a single reusable node.

        Args:
            nodes: List of FunctionNodes representing the subgraph to compress.
            name: The name to give to the compressed node.

        Returns:
            A FunctionNode representing the compressed subgraph.
        """
        pass

    @abstractmethod
    def available_functions(self) -> Dict[str, FunctionNode]:
        """
        Returns a dictionary of available function nodes.

        Includes standard nodes (ReLU, Sigmoid, Add) and previously compressed graphs.

        Returns:
            A dictionary mapping node names to FunctionNode instances.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Checks if the graph is empty.

        Returns:
            True if the graph is empty, False otherwise.
        """
        pass


class SimpleComposer(GraphFunctionComposer):  # Start by adding a single node
    def __init__(self):
        super().__init__()  # Important: Call the superclass constructor
        self.graph: Dict[str, FunctionNode] = {}  # Dictionary to store nodes

    @overload
    def add_function(self, node: FunctionNode) -> FunctionNode:
        ...

    @overload
    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode]) -> FunctionNode:
        ...

    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode] = None) -> FunctionNode:
        if input_nodes is None:
            if self.is_empty():
              self.graph[node.name] = node
              return node
            else:
              raise ValueError("Input nodes must be provided for subsequent nodes")
        else:
            if not self.is_empty():
              for input_node in input_nodes:
                  if input_node.name not in self.graph:
                      raise ValueError(f"Input node '{input_node.name}' not found in graph.")
                  node.add_input(self.graph[input_node.name])
              self.graph[node.name] = node
              return node
            else:
              raise ValueError("First node cannot have input nodes")

    def get_function(self, name: str) -> FunctionNode:
        if name not in self.graph:
            raise ValueError(f"Node '{name}' not found.")
        return self.graph[name]

    def compress_graph(self, nodes: List[FunctionNode], name: str) -> FunctionNode:
        raise NotImplementedError("Graph compression not implemented yet.")

    def available_functions(self) -> Dict[str, FunctionNode]:
        return self.graph.copy()

    def is_empty(self) -> bool:
        return not self.graph

    def execute(self, input_data):
        if not self.graph:
            raise ValueError("Graph is empty.")

        # For this initial version, assume only one node in the graph
        node = list(self.graph.values())[0] # Get the only node
        node.build(input_data.shape)  # Build the node using the shape of the input data
        return node(input_data)       # Execute the node with the input data
