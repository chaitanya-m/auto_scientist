# graph/composer.py
import keras
from collections import deque
from graph.node import SingleNeuron, InputNode  # Import blueprint node types

class GraphComposer:
    """
    Assembles a final Keras model from blueprint nodes.
    Each node is a specification of a Keras transformation.
    When build() is called, the composer wires together nodes via the Functional API.
    This version only supports a single input.
    """
    def __init__(self):
        self.nodes = {}         # Mapping from node name to GraphNode blueprint object.
        self.connections = {}   # Mapping from node name -> list of (parent node name, merge mode) tuples.
        self.input_node_name = None  # Name of the single input node.
        self.output_node_names = None  # List of output node names.
        self.keras_model = None # Keras model built using the Functional API for this graph

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")
        self.nodes[node.name] = node

    def set_input_node(self, node_name):
        # Ensure only a single input node is provided.
        if isinstance(node_name, list):
            if len(node_name) != 1:
                raise ValueError("Only a single input node is allowed.")
            node_name = node_name[0]
        if node_name not in self.nodes:
            raise ValueError(f"Input node '{node_name}' not found.")
        if not isinstance(self.nodes[node_name], InputNode):
            raise ValueError(f"Node '{node_name}' is not an InputNode.")
        self.input_node_name = node_name

    def set_output_node(self, node_names):
        if isinstance(node_names, list):
            if not node_names:
                raise ValueError("At least one output node must be set.")
            for n in node_names:
                if n not in self.nodes:
                    raise ValueError(f"Output node '{n}' not found.")
            self.output_node_names = node_names
        else:
            if node_names not in self.nodes:
                raise ValueError(f"Output node '{node_names}' not found.")
            self.output_node_names = [node_names]

    def connect(self, from_node, to_node, merge_mode='concat'):
        if from_node not in self.nodes:
            raise ValueError(f"From-node '{from_node}' not found.")
        if to_node not in self.nodes:
            raise ValueError(f"To-node '{to_node}' not found.")
        self.connections.setdefault(to_node, []).append((from_node, merge_mode))

    def build(self):
        """
        Assembles and returns the final Keras model.
        Wires the graph using the Keras Functional API.
        Assumes exactly one input node.
        """
        if self.input_node_name is None or self.output_node_names is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")

        print("\nBuilding blueprint graph\n")

        # Create the single input layer.
        input_node = self.nodes[self.input_node_name]
        input_layer = keras.layers.Input(shape=input_node.input_shape, name=self.input_node_name)

        # Store the output of the input node.
        node_output_tensors = {}
        node_output_tensors[self.input_node_name] = input_node.apply(input_layer)

        # Topologically sort the nodes.
        order = self._topological_sort()
        for node_name in order:
            if node_name in node_output_tensors:
                continue
            node = self.nodes[node_name]
            parent_infos = self.connections.get(node_name, [])
            if not parent_infos:
                raise ValueError(f"Non-InputNode '{node_name}' has no parents.")
            parent_tensors = []
            merge_modes = []
            for parent_name, merge_mode in parent_infos:
                parent_tensors.append(node_output_tensors[parent_name])
                merge_modes.append(merge_mode)
            if len(parent_tensors) == 1:
                combined = parent_tensors[0]
            else:
                if all(m == "add" for m in merge_modes):
                    combined = keras.layers.Add(name=f"{node_name}_add")(parent_tensors)
                elif all(m == "concat" for m in merge_modes):
                    combined = keras.layers.Concatenate(name=f"{node_name}_concat")(parent_tensors)
                else:
                    raise ValueError(f"Mixed merge modes for node '{node_name}'")
            node_output_tensors[node_name] = node.apply(combined)

        outputs = [node_output_tensors[name] for name in self.output_node_names]

        # Build the final model with a single input.
        self.keras_model = keras.models.Model(
            inputs=input_layer,
            outputs=outputs if len(outputs) > 1 else outputs[0],
            name="GraphModel"
        )
        return self.keras_model

    def _topological_sort(self):
        """
        Performs a topological sort on the graph to determine a valid computation order.
        Ensures that each node is processed only after all its dependencies (parents)
        have been processed.
        Returns:
            order (list): A list of node names sorted in topological order.
        Raises:
            ValueError: If a cycle is detected or the graph is disconnected.
        """
        in_degree = {name: 0 for name in self.nodes}
        for child, parent_infos in self.connections.items():
            in_degree[child] += len(parent_infos)
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child, parent_infos in self.connections.items():
                if current in [pi[0] for pi in parent_infos]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected or graph is disconnected.")
        return order

    def save_subgraph(self, filepath):
        """
        Saves a subgraph with a single input.
        Since only single inputs are supported, the subgraph is saved to the provided filepath
        (with the extension replaced to ".keras").
        """
        if self.keras_model is None:
            raise ValueError("Build the model before saving subgraph.")

        input_tensor = self.keras_model.input
        orig_name = input_tensor.name.split(":")[0]
        new_input = keras.layers.Input(
            shape=input_tensor.shape[1:],
            name=f"subgraph_{orig_name}"
        )
        new_output = self.keras_model(new_input)
        new_model = keras.models.Model(new_input, new_output)
        new_model.set_weights(self.keras_model.get_weights())
        new_model.save(filepath.replace(".h5", ".keras"))
