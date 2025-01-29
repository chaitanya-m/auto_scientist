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
        self.connections: Dict[str, List[str]] = {}  # Dictionary to store node connections

    @overload
    def add_function(self, node: FunctionNode) -> FunctionNode:
        ...

    @overload
    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode]) -> FunctionNode:
        ...


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

    def add_function(self, node: FunctionNode, input_nodes: List[FunctionNode] = None) -> FunctionNode:
        if input_nodes is None:  # First node
            if self.is_empty():
                self.graph[node.name] = node
                self.connections[node.name] = []  # No inputs for the first node
                return node
            else:
                raise ValueError("Input nodes must be provided for subsequent nodes.")
        else:  # Subsequent nodes
            if not self.is_empty():
                for input_node in input_nodes:
                    if input_node.name not in self.graph:
                        raise ValueError(f"Input node '{input_node.name}' not found in graph.")
                self.graph[node.name] = node
                self.connections[node.name] = [n.name for n in input_nodes] # Store connections
                return node
            else:
                raise ValueError("First node cannot have input nodes.")

    def execute(self, input_data):
        if not self.graph:
            raise ValueError("Graph is empty.")

        # Build all nodes first
        for node in self.graph.values():
            input_shape = input_data.shape if not self.connections.get(node.name) else self.graph[self.connections[node.name][0]].output_shape
            node.build(input_shape)

        # Execute the graph
        outputs = {"input": input_data}  # Initialize outputs with input data
        for node_name, node in self.graph.items():
            input_names = self.connections.get(node_name, [])  # Get input node names
            inputs = [outputs[name] for name in input_names] if input_names else [input_data]
            outputs[node_name] = node(inputs)  # Execute and store the output

        # Return the output of the last node
        return outputs[list(self.graph.keys())[-1]]