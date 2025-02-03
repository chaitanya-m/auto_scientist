import keras
from collections import deque

class GraphComposer:
    """
    Assembles a final Keras model from GraphNode blueprints.
    Nodes and their connections are defined as blueprints (with no immediate instantiation).
    When build() is called, the composer traverses the blueprint graph and
    uses the Keras Functional API to build a unified, vanilla Keras model.
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
        """
        Designate the input node(s). Accepts either a string or a list of strings.
        """
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
        """
        Designate the output node(s). Accepts either a string or a list of strings.
        """
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
        """
        Connects the output of one node to the input of another.
        """
        if from_node not in self.nodes:
            raise ValueError(f"From-node '{from_node}' not found.")
        if to_node not in self.nodes:
            raise ValueError(f"To-node '{to_node}' not found.")
        self.connections.setdefault(to_node, []).append(from_node)

    def build(self, input_shape):
        """
        Assembles and returns the final Keras model.
        This method creates a single global Input layer and then, based on the blueprint,
        wires together all nodes via the Functional API.
        """
        if self.input_node_names is None or self.output_node_names is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")

        # Create the global input tensor.
        global_input = keras.layers.Input(shape=input_shape, name="global_input")

        # For each designated input node, build its branch using the global input.
        node_outputs = {}
        for in_name in self.input_node_names:
            node = self.nodes[in_name]
            node_outputs[in_name] = node.apply(global_input)

        # Process remaining nodes in topological order.
        order = self._topological_sort()
        for node_name in order:
            if node_name in node_outputs:
                # Already processed (i.e. designated as an input node).
                continue
            node = self.nodes[node_name]
            parent_names = self.connections.get(node_name, [])
            if not parent_names:
                # If no parents are specified, default to using the global input.
                parent_tensor = global_input
            elif len(parent_names) == 1:
                parent_tensor = node_outputs[parent_names[0]]
            else:
                # For multiple parent nodes, concatenate their outputs along the last axis.
                parent_tensors = [node_outputs[p] for p in parent_names]
                parent_tensor = keras.layers.Concatenate(name=f"{node_name}_concat")(parent_tensors)
            node_outputs[node_name] = node.apply(parent_tensor)

        # Collect the outputs from the designated output nodes.
        outputs = [node_outputs[name] for name in self.output_node_names]
        self.keras_model = keras.models.Model(
            inputs=global_input,
            outputs=outputs if len(outputs) > 1 else outputs[0],
            name="GraphModel"
        )
        return self.keras_model

    def _topological_sort(self):
        """
        Computes a topological ordering of the nodes based on the connections.
        Raises an error if a cycle is detected.
        """
        in_degree = {name: 0 for name in self.nodes}
        for child, parents in self.connections.items():
            in_degree[child] += len(parents)
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
