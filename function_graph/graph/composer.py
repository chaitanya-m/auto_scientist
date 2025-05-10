# graph/composer.py
import keras
from collections import deque
from graph.node import SingleNeuron, InputNode, SubGraphNode  # Import blueprint node types
import random
import copy
import uuid
import hashlib  # Ensure hashlib is imported

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

    def remove_connection(self, from_node, to_node, merge_mode=None):
        """
        Removes a connection from 'from_node' to 'to_node'.
        If merge_mode is provided, only removes connections with that merge_mode.
        Otherwise, it removes all connections from 'from_node' to 'to_node'.
        """
        if to_node not in self.connections:
            raise ValueError(f"No connections found for target node '{to_node}'")
        
        original_length = len(self.connections[to_node])
        if merge_mode is None:
            self.connections[to_node] = [
                connection for connection in self.connections[to_node] if connection[0] != from_node
            ]
        else:
            self.connections[to_node] = [
                connection for connection in self.connections[to_node]
                if not (connection[0] == from_node and connection[1] == merge_mode)
            ]
        if len(self.connections[to_node]) == original_length:
            raise ValueError(f"No connection found from '{from_node}' to '{to_node}' with merge mode '{merge_mode}'")

    def build(self):
        """
        Assembles and returns the final Keras model.
        Wires the graph using the Keras Functional API.
        Assumes exactly one input node.
        """
        if self.input_node_name is None or self.output_node_names is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")

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
                parent_tensor = node_output_tensors[parent_name]
                # If the parent node is a SubGraphNode, insert an adapter layer.
                if isinstance(self.nodes[parent_name], SubGraphNode):
                    # Assume the output dimension remains the same.
                    dim = int(parent_tensor.shape[-1])
                    adapter_name = f"fast_adapter_{parent_name}_{uuid.uuid4().hex[:3]}"
                    parent_tensor = keras.layers.Dense(dim, activation="linear", name=adapter_name)(parent_tensor)
                parent_tensors.append(parent_tensor)
                merge_modes.append(merge_mode)

            if len(parent_tensors) == 1:
                combined = parent_tensors[0]
            else:
                unique_suffix = uuid.uuid4().hex[:6]

                if all(m == "add" for m in merge_modes):
                    combined = keras.layers.Add(name=f"{node_name}_add_{unique_suffix}")(parent_tensors)
                elif all(m == "concat" for m in merge_modes):
                    combined = keras.layers.Concatenate(name=f"{node_name}_concat_{unique_suffix}")(parent_tensors)
                else:
                    raise ValueError(f"Mixed merge modes for node '{node_name}'")

            node_output_tensors[node_name] = node.apply(combined)

        outputs = [node_output_tensors[name] for name in self.output_node_names]

        # Build the final model with a single input.
        unique_model_name = "GraphModel_" + str(uuid.uuid4())
        self.keras_model = keras.models.Model(
            inputs=input_layer,
            outputs=outputs if len(outputs) > 1 else outputs[0],
            name=unique_model_name
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


    def safe_insert_subgraph(self, sub_node, input_name="input", output_name="output"):
        """
        Attempt to insert `sub_node` into this composer without creating cycles.
        Returns True and mutates self if successful, False otherwise.
        """
        import copy
        trial = copy.deepcopy(self)

        # Add the subgraph node
        try:
            trial.add_node(sub_node)
        except ValueError:
            return False

        # Rewire input→sub→output
        try:
            trial.disconnect(input_name, output_name)
        except Exception:
            pass
        trial.connect(input_name, sub_node.name)
        trial.connect(sub_node.name, output_name)

        # Test for cycles or disconnects
        try:
            trial.build()
        except ValueError:
            return False

        # Commit: copy trial’s nodes and connections back into self
        self.nodes = trial.nodes
        self.connections = trial.connections
        return True



class GraphTransformer:
    """
    Provides high-level transformations on a GraphComposer,
    such as inserting a learned abstraction node in the graph.
    """
    def __init__(self, composer: GraphComposer):
        self.composer = composer

    def get_composer(self):
        """
        Returns the composer instance - this is, for instance, useful when hashing.
        """
        return self.composer

    def add_abstraction_node(
        self,
        abstraction_node,
        chosen_subset: list[str],
        outputs: list[str],
        remove_prob: float = 1.0
    ):
        """
        Adds a learned abstraction node to the graph. For each node in chosen_subset,
        connect it to the new abstraction node, then connect the abstraction node to
        all output nodes. Finally, remove direct connections from chosen_subset to
        each output node with probability remove_prob, then rebuild the Keras model.

        Parameters
        ----------
        abstraction_node : SubGraphNode
            The learned abstraction node to be inserted (must have a unique name).
        chosen_subset : list of str
            Names of nodes (excluding output nodes) to feed into the abstraction node.
        outputs : list of str
            Names of the current output nodes in the graph.
        remove_prob : float
            Probability (0 to 1) of removing the direct node→output connection for each
            (node in chosen_subset, output) pair:
            - 1.0 => remove all direct connections
            - 0.0 => keep all direct connections
            - 0.5 => remove about half on average
        """
        composer = self.composer

        # Always deep-copy the abstraction node to ensure a fresh instance.
        new_abstraction = copy.deepcopy(abstraction_node)

        # Generate a short random suffix (e.g. 4 hex characters).
        unique_suffix = str(uuid.uuid4().hex[:4])

        # Rename the top-level node
        new_abstraction.name = f"{abstraction_node.name}_{unique_suffix}"

        # Rebuild the bypass model from scratch so that all internal layer names are freshly generated.
        new_input = keras.layers.Input(
            shape=abstraction_node.model.input.shape[1:], 
            name=f"{new_abstraction.name}_bypass_input"
        )
        # Pass the new input through the original model to get the output.
        x = abstraction_node.model(new_input)
        new_bypass_model_name = f"{new_abstraction.name}_bypass_model"
        new_abstraction.bypass_model = keras.models.Model(new_input, x, name=new_bypass_model_name)

        # Rename all keras layers in the bypass model
        for layer in new_abstraction.bypass_model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            unique_layer_name = f"{layer.name}_{unique_suffix}"
            try:
                layer._name = unique_layer_name
            except Exception as e:
                print(f"DEBUG: Could not update layer name for {layer.name} due to {e}")

        # Add the new, uniquely-named abstraction node.
        composer.add_node(new_abstraction)
        # Connect each node in the chosen subset to the new abstraction node.
        for node_name in chosen_subset:
            composer.connect(node_name, new_abstraction.name, merge_mode='concat')
        # Connect the new abstraction node to each output.
        for out_name in outputs:
            composer.connect(new_abstraction.name, out_name, merge_mode='concat')
        # Remove direct connections from chosen_subset to outputs with probability remove_prob.
        for node_name in chosen_subset:
            for out_name in outputs:
                if random.random() < remove_prob:
                    try:
                        composer.remove_connection(node_name, out_name)
                    except ValueError:
                        pass

        # Clear any previously built model to start fresh.
        composer.keras_model = None

        new_model = composer.build()

        new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return new_model


# ---------------------------------------------------------------------------
# Hashing functionality refactored into a dedicated class
# ---------------------------------------------------------------------------

class GraphHasher:
    """
    Provides methods to generate a canonical string representation and a hash for a GraphComposer's structure.
    
    This functionality:
      1. Serializes graph nodes by sorting them, extracting their key parameters as defined below.
      2. Serializes graph connections by sorting and formatting them.
      3. Combines both serialized representations into a canonical form and computes an MD5 hash.
    """
    @staticmethod
    def canonical_representation(composer):
        """
        Generates a canonical string representation of a GraphComposer's structure.
    
        This function:
          1. Iterates over the composer’s nodes (a dictionary mapping node names to node objects),
             sorting them by node name.
          2. For each node, it extracts the node type (using __class__.__name__) and key parameters
             that describe the node's behavior (for example, input_shape for an InputNode, activation
             (and optionally units) for a SingleNeuron or DenseNode). This information is concatenated
             into a string for each node.
          3. Iterates over the composer’s connections (a dictionary mapping a target node name to a list
             of (parent_node, merge_mode) tuples), sorts them by target node name and then by the parent's
             name and merge_mode, and serializes them into a string.
          4. Concatenates the node and connection strings (with delimiters) to produce a complete,
             canonical representation of the graph structure.
    
        Args:
            composer (GraphComposer): The graph composer whose structure is to be hashed.
    
        Returns:
            str: A canonical string representation of the graph.
        """
        # Serialize nodes: sort by node name to ensure consistent ordering.
        node_strs = []
        for node_name in sorted(composer.nodes.keys()):
            node = composer.nodes[node_name]
            # Determine key parameters based on node type.
            if hasattr(node, 'input_shape'):
                params = f"input_shape={node.input_shape}"
            elif hasattr(node, 'activation') and hasattr(node, 'units'):
                # For nodes like DenseNode that specify units and activation.
                params = f"units={node.units},activation={node.activation}"
            elif hasattr(node, 'activation'):
                # For nodes like SingleNeuron.
                params = f"activation={node.activation}"
            else:
                # Fallback if no key parameters are defined.
                params = "N/A"
            node_str = f"{node_name}:{node.__class__.__name__}({params})"
            node_strs.append(node_str)
        nodes_part = ";".join(node_strs)
    
        # Serialize connections: sort keys and each parent's entry for consistency.
        connection_strs = []
        for child_name in sorted(composer.connections.keys()):
            parent_list = composer.connections[child_name]
            # Sort parent's by name and merge_mode.
            sorted_parents = sorted(parent_list, key=lambda x: (x[0], x[1]))
            for parent, merge_mode in sorted_parents:
                connection_strs.append(f"{parent}->{child_name}[merge={merge_mode}]")
        # Sort the connection strings themselves.
        connections_part = ";".join(sorted(connection_strs))
    
        # Combine the nodes and connections into one canonical string.
        full_repr = f"Nodes:{nodes_part}|Connections:{connections_part}"
        return full_repr

    @staticmethod
    def hash(composer):
        """
        Returns a hash for the given graph.
    
        The graph is first serialized into a canonical string (via canonical_representation)
        capturing its node types, key parameters, and connections, and then hashed using MD5.
    
        Args:
            composer (GraphComposer): The graph composer to hash.
    
        Returns:
            str: A fixed-length MD5 hash of the graph's canonical representation.
        """
        canonical_str = GraphHasher.canonical_representation(composer)
        return hashlib.md5(canonical_str.encode('utf-8')).hexdigest()
