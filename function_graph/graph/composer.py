# tests/test_composer.py
import keras
from collections import deque
from graph.node import SingleNeuron, InputNode  # Import the blueprint node types

class GraphComposer:
    """
    Assembles a final Keras model from blueprint nodes.
    Each node is a specification of a Keras transformation.
    When build() is called, the composer wires together nodes via the Functional API.
    """
    def __init__(self):
        self.nodes = {}         # Dictionary mapping from node name to GraphNode blueprint object.
        self.connections = {}   # Dictionary mapping node name -> list of parent node names.
        self.input_node_names = None  # List of input node names.
        self.output_node_names = None  # List of output node names.
        self.keras_model = None # keras_model built using functional API for this graph

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")
        self.nodes[node.name] = node

    def set_input_node(self, node_names):
        if isinstance(node_names, list):
            for n in node_names:
                if n not in self.nodes:
                    raise ValueError(f"Input node '{n}' not found.")
                if not isinstance(self.nodes[n], InputNode):
                    raise ValueError(f"Node '{n}' is not an InputNode.")
            self.input_node_names = node_names
        else:
            if node_names not in self.nodes:
                raise ValueError(f"Input node '{node_names}' not found.")
            if not isinstance(self.nodes[node_names], InputNode):
                raise ValueError(f"Node '{node_names}' is not an InputNode.")
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
        """
        if self.input_node_names is None or self.output_node_names is None:
            raise ValueError("Both input and output nodes must be set before building the graph.")

        print("\nBuilding blueprint graph\n")

        # Collect Input layers as a dictionary for explicit structure
        input_layers = {}
        for input_name in self.input_node_names:
            node = self.nodes[input_name]
            if not isinstance(node, InputNode):
                raise ValueError(f"Node '{input_name}' is not an InputNode.")
            input_layer = keras.layers.Input(
                shape=node.input_shape, 
                name=input_name  # Keep names exactly as defined
            )
            input_layers[input_name] = input_layer

        node_output_tensors = {}
        
        # Process InputNodes using dictionary keys
        for input_name in self.input_node_names:
            node = self.nodes[input_name]
            node_output_tensors[input_name] = node.apply(input_layers[input_name])

        # Convert to ordered list to preserve input order
        model_inputs = [input_layers[name] for name in self.input_node_names]

        # Process remaining nodes (same as before)
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

        # Explicitly define input structure using a dictionary
        self.keras_model = keras.models.Model(
            inputs={name: input_layers[name] for name in self.input_node_names},
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
        in_degree = {name: 0 for name in self.nodes}
        
        # Calculate initial in-degree for each node
        for child, parent_infos in self.connections.items():
            in_degree[child] += len(parent_infos)  # Each parent_info contributes to in-degree
        
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []
        
        while queue:
            current = queue.popleft()
            order.append(current)
            
            # Check all nodes to see if they depend on the current node
            for child, parent_infos in self.connections.items():
                # Extract parent names from (parent_name, merge_mode) tuples
                parent_names = [pi[0] for pi in parent_infos]
                if current in parent_names:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected or graph is disconnected.")
        
        return order

    

    def save_subgraph(self, filepath):
        """
        Saves a subgraph as a standalone Keras model that can be reloaded later.
        Handles input structure matching exactly with original model.
        """
        if self.keras_model is None:
            raise ValueError("Build the model before saving subgraph.")

        # Get original input structure
        original_inputs = self.keras_model.input

        # Create matching new inputs
        if isinstance(original_inputs, dict):
            # Handle dictionary inputs
            new_inputs = {
                name: keras.layers.Input(shape=tensor.shape[1:], name=name)
                for name, tensor in original_inputs.items()
            }
            tensor_map = new_inputs
        elif isinstance(original_inputs, list):
            # Handle list inputs
            new_inputs = [
                keras.layers.Input(shape=tensor.shape[1:])
                for tensor in original_inputs
            ]
            tensor_map = dict(zip(original_inputs, new_inputs))
        else:
            # Handle single tensor input
            new_inputs = keras.layers.Input(
                shape=original_inputs.shape[1:], 
                name=original_inputs.name.split(":")[0]  # Get base name
            )
            tensor_map = {original_inputs: new_inputs}

        # Clone model with matching structure
        cloned_model = keras.models.clone_model(
            self.keras_model,
            input_tensors=tensor_map
        )
        cloned_model.set_weights(self.keras_model.get_weights())
        cloned_model.save(filepath.replace(".h5", ".keras"))