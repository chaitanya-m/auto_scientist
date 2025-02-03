# composer.py
import keras
from collections import deque

class GraphComposer:
    """
    Builds a computation graph from GraphNode instances. 
    Nodes and their connections are specified independently of Keras.
    When build() is called, the composer assembles a complete Keras model
    using the Functional API.
    """
    def __init__(self):
        self.nodes = {}         # Mapping from node name to GraphNode instance
        self.connections = {}   # Mapping: node name -> list of parent node names
        self.input_node_name = None
        self.output_node_name = None
        self.keras_model = None

    def add_node(self, node):
        """
        Adds a node to the graph.
        
        Args:
            node: An instance of GraphNode.
            
        Raises:
            ValueError: If a node with the same name already exists.
        """
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")
        self.nodes[node.name] = node

    def set_input_node(self, node_name):
        """
        Designates the input node.
        
        Args:
            node_name (str): The name of the node to use as the input.
            
        Raises:
            ValueError: If the specified node is not found.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Input node '{node_name}' not found.")
        self.input_node_name = node_name

    def set_output_node(self, node_name):
        """
        Designates the output node.
        
        Args:
            node_name (str): The name of the node to use as the output.
            
        Raises:
            ValueError: If the specified node is not found.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Output node '{node_name}' not found.")
        self.output_node_name = node_name

    def connect(self, from_node, to_node):
        """
        Connects the output of one node to the input of another.
        
        Args:
            from_node (str): The name of the node whose output is to be connected.
            to_node (str): The name of the node that receives the connection.
            
        Raises:
            ValueError: If either node is not found.
        """
        if from_node not in self.nodes:
            raise ValueError(f"From-node '{from_node}' not found.")
        if to_node not in self.nodes:
            raise ValueError(f"To-node '{to_node}' not found.")
        self.connections.setdefault(to_node, []).append(from_node)

    def build(self, input_shape):
        """
        Assembles the complete Keras model using the Functional API.
        
        Args:
            input_shape (tuple): Global input shape (excluding the batch dimension).
            
        Returns:
            A Keras model representing the assembled graph.
            
        Raises:
            ValueError: If input and/or output nodes are not set.
        """
        if self.input_node_name is None or self.output_node_name is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")
        
        # Create the global input tensor.
        global_input = keras.layers.Input(shape=input_shape, name="global_input")
        
        # Build and apply the input node.
        input_node = self.nodes[self.input_node_name]
        input_node.build_node(input_shape)
        node_outputs = {self.input_node_name: input_node.apply(global_input)}
        
        # Process remaining nodes in topological order.
        order = self._topological_sort()
        for node_name in order:
            if node_name in node_outputs:
                # Already processed (for example, the designated input node).
                continue
            node = self.nodes[node_name]
            parent_names = self.connections.get(node_name, [])
            if not parent_names:
                # If no parents, default to the global input.
                parent_tensor = global_input
            elif len(parent_names) == 1:
                parent_tensor = node_outputs[parent_names[0]]
            else:
                # For multiple parents, concatenate their outputs along the last axis.
                parent_tensors = [node_outputs[p] for p in parent_names]
                parent_tensor = keras.layers.Concatenate(name=f"{node_name}_concat")(parent_tensors)
            
            # Infer input shape from the parent's output (excluding batch dimension).
            # Depending on the environment, parent_tensor.shape might be a tuple or a TensorShape.
            if hasattr(parent_tensor.shape, "as_list"):
                shape_list = parent_tensor.shape.as_list()
            else:
                shape_list = parent_tensor.shape
            node_input_shape = tuple(shape_list[1:])
            
            node.build_node(node_input_shape)
            node_outputs[node_name] = node.apply(parent_tensor)
        
        # Assemble the final model.
        self.keras_model = keras.models.Model(
            inputs=global_input,
            outputs=node_outputs[self.output_node_name],
            name="GraphModel"
        )
        return self.keras_model

    def _topological_sort(self):
        """
        Computes a topological ordering of the nodes.
        
        Returns:
            A list of node names in topological order.
            
        Raises:
            ValueError: If a cycle is detected.
        """
        # Compute in-degrees for each node.
        in_degree = {name: 0 for name in self.nodes}
        for child, parents in self.connections.items():
            in_degree[child] += len(parents)
        
        # Start with all nodes that have zero in-degree.
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child, parents in self.connections.items():
                if current in parents:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected or graph is disconnected.")
        return order
