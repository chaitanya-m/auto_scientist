import keras
from collections import deque
from graph.node import SingleNeuron, InputNode  # Import the blueprint node type

class GraphComposer:
    """
    Assembles a final Keras model from blueprint nodes.
    Each node is a specification of a Keras transformation.
    When build() is called, the composer either collapses a simple (dense) graph 
    into a single Dense layer or wires together nodes via the Functional API.
    """
    def __init__(self):
        self.nodes = {}         # Mapping from node name to GraphNode blueprint.
        self.connections = {}   # Mapping: node name -> list of parent node names.
        self.input_node_names = None  # List of input node names.
        self.output_node_names = None  # List of output node names.
        self.keras_model = None

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")
        self.nodes[node.name] = node

    def set_input_node(self, node_names):
        if isinstance(node_names, list):
            for n in node_names:
                if n not in self.nodes:
                    raise ValueError(f"Input node '{n}' not found.")
            self.input_node_names = node_names
        else:
            if node_names not in self.nodes:
                raise ValueError(f"Input node '{node_names}' not found.")
            self.input_node_names = [node_names]

    def set_output_node(self, node_names):
        if isinstance(node_names, list):
            for n in node_names:
                if n not in self.nodes:
                    raise ValueError(f"Output node '{n}' not found.")
            self.output_node_names = node_names
        else:
            if node_names not in self.nodes:
                raise ValueError(f"Output node '{node_names}' not found.")
            self.output_node_names = [node_names]

    def connect(self, from_node, to_node):
        if from_node not in self.nodes:
            raise ValueError(f"From-node '{from_node}' not found.")
        if to_node not in self.nodes:
            raise ValueError(f"To-node '{to_node}' not found.")
        self.connections.setdefault(to_node, []).append(from_node)

    def build(self, input_shape):
        """
        Assembles and returns the final Keras model.
        If the blueprint graph is "collapsible" (i.e. only input and output nodes,
        all are SingleNeuron with linear activation, and each output node receives
        input from all input nodes), then build a single Dense layer.
        Otherwise, wire the graph using the Keras Functional API.
        """
        if self.input_node_names is None or self.output_node_names is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")

        print("\nBuilding blueprint graph\n")
        global_input = keras.layers.Input(shape=input_shape, name="global_input")
        node_outputs = {}
        # For each designated input node, call its blueprint apply() using the global input.

        # Input nodes pass inputs directly, without transformations
        for in_name in self.input_node_names:
            node = self.nodes[in_name]
            if isinstance(node, InputNode):  # Input nodes should not modify data
                node_outputs[in_name] = global_input  # Directly forward input
            else:
                node_outputs[in_name] = node.apply(global_input)  # Apply transformation

        # Process remaining nodes in topological order.
        order = self._topological_sort()
        for node_name in order:
            if node_name in node_outputs:
                continue
            node = self.nodes[node_name]
            parent_names = self.connections.get(node_name, [])
            if not parent_names:
                parent_tensor = global_input
            elif len(parent_names) == 1:
                parent_tensor = node_outputs[parent_names[0]]
            else:
                parent_tensors = [node_outputs[p] for p in parent_names]
                parent_tensor = keras.layers.Concatenate(name=f"{node_name}_concat")(parent_tensors)
            node_outputs[node_name] = node.apply(parent_tensor)

        outputs = [node_outputs[name] for name in self.output_node_names]
        self.keras_model = keras.models.Model(
            inputs=global_input,
            outputs=outputs if len(outputs) > 1 else outputs[0],
            name="GraphModel"
        )
        return self.keras_model

    def _topological_sort(self):
        """
        Performs a topological sort on the graph to determine a valid computation order.
        
        This ensures that each node is processed only after all its dependencies (parents)
        have been processed.
        
        Returns:
            order (list): A list of node names sorted in topological order.

        Raises:
            ValueError: If a cycle is detected or the graph is disconnected.
        """

        in_degree = {name: 0 for name in self.nodes} # Initialize in-degree for each node.
        for child, parents in self.connections.items(): # Count incoming edges for each node.
            in_degree[child] += len(parents) 
        queue = deque([name for name, deg in in_degree.items() if deg == 0])  # Nodes with no dependencies.
        order = [] # Stores sorted nodes.
        while queue:
            current = queue.popleft()  # Process next node with no remaining dependencies.
            order.append(current)  # Add it to the sorted order.
            for child, parents in self.connections.items():
                if current in parents:
                    in_degree[child] -= 1 # Reduce child's in-degree.
                    if in_degree[child] == 0:
                        queue.append(child) # Add child if it has no more dependencies.
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected or graph is disconnected.")  # Ensure all nodes are processed.
        return order  # Return nodes in topological order.
