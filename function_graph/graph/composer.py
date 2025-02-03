import keras
from collections import deque
from graph.node import SingleNeuron  # Import the blueprint node type

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

        # --- Check for collapsibility ---
        # We define the graph as collapsible if:
        # 1. The set of nodes equals the union of input and output nodes.
        # 2. Every node is an instance of SingleNeuron with activation "linear".
        # 3. For every output node, the list of parent connections exactly equals the list of input nodes.
        all_node_names = set(self.nodes.keys())
        blueprint_nodes = set(self.input_node_names) | set(self.output_node_names)
        if all_node_names == blueprint_nodes:
            # Check that each node is a SingleNeuron with linear activation.
            if all(isinstance(self.nodes[n], SingleNeuron) and self.nodes[n].activation == "linear" for n in all_node_names):
                # Check that every output node receives connections from all input nodes.
                collapsible = True
                for out_node in self.output_node_names:
                    # If no connections are specified, assume that the blueprint intends the output node
                    # to use the global input. In that case, if there is more than one input node, we expect a full connection.
                    parents = self.connections.get(out_node, [])
                    if set(parents) != set(self.input_node_names):
                        collapsible = False
                        break
                if collapsible:
                    # We can collapse the graph into a single Dense layer.
                    global_input = keras.layers.Input(shape=input_shape, name="global_input")
                    # Create a Dense layer with units equal to the number of output nodes.
                    dense = keras.layers.Dense(len(self.output_node_names), activation="linear", name="dense_collapse")(global_input)
                    self.keras_model = keras.models.Model(inputs=global_input, outputs=dense, name="CollapsedGraphModel")
                    return self.keras_model
        # --- End collapsibility check ---

        # Fallback: build the full graph using the blueprint connections.

        print("\nBuilding blueprint graph\n")
        global_input = keras.layers.Input(shape=input_shape, name="global_input")
        node_outputs = {}
        # For each designated input node, call its blueprint apply() using the global input.
        for in_name in self.input_node_names:
            node = self.nodes[in_name]
            node_outputs[in_name] = node.apply(global_input)
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
