# composer.py
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Dict, overload
from function_graph.node import FunctionNode, TrainableNode

# ... (FunctionNode, Trainable, TrainableNode, FixedActivation, 
#      FixedSigmoid, FixedReLU, ReLU, Sigmoid classes remain unchanged)

class FunctionGraph:  # The core graph structure (separate class)
    def __init__(self):
        self.nodes: Dict[str, FunctionNode] = {}
        self.input_node = None
        self.output_node = None
        self.connections: Dict[str, List[str]] = {}  # Store connections
        self.input_shape = None  # Initialize input_shape

    def add(self, node):
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists.")

        self.nodes[node.name] = node

        if self.input_shape is not None and node == self.input_node:  # Correctly set input shape for the first node
            node.build(self.input_shape)  # Build the input node immediately

    def connect(self, from_node_name, to_node_name):
        from_node = self.nodes[from_node_name]
        to_node = self.nodes[to_node_name]

        if to_node.input_shape is None:  # Build the to_node if it hasn't been built yet.
            if from_node == self.input_node and self.input_shape is not None:  # Correctly handle the first connection when input shape is rank 1
                to_node.build(self.input_shape)
            else:
                to_node.build(from_node.output_shape)  # Build the to_node based on input from from_node

        if from_node.output_shape is None and from_node != self.input_node:
            raise ValueError(f"Node '{from_node_name}' must be built before connecting.")

        if to_node_name not in self.connections:
            self.connections[to_node_name] = []

        self.connections[to_node_name].append(from_node_name)  # store connection

    def set_input(self, node_name):
        self.input_node = self.nodes[node_name]

    def set_output(self, node_name):
        self.output_node = self.nodes[node_name]

    def forward(self, inputs):
        node_outputs = {self.input_node.name: inputs}  # Initialize with input

        for to_node_name, to_node in self.nodes.items():
            if to_node == self.input_node:
                continue  # Input node already handled

            input_names = self.connections.get(to_node_name, [])
            node_inputs = [node_outputs[name] for name in input_names]

            if len(node_inputs) == 1:
                node_inputs = node_inputs[0]  # if only one input, unpack the list

            node_outputs[to_node_name] = to_node(node_inputs)

        return node_outputs[self.output_node.name]

    def train(self, inputs, targets, optimizer, loss_function, epochs=1, verbose=0):  # Add optimizer and loss_function
        if not hasattr(self, 'optimizer') or not hasattr(self, 'loss_function'):
            raise ValueError("Graph must be compiled before training.")

        inputs = tf.cast(inputs, tf.float32)  # Type casting for inputs

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                outputs = self.forward(inputs)  # Forward pass through the graph
                loss = loss_function(targets, outputs)  # Calculate the loss

            trainable_variables = []
            for node in self.nodes.values():
                if isinstance(node, TrainableNode):
                    trainable_variables.extend(node.trainable_variables)

            gradients = tape.gradient(loss, trainable_variables)

            # Handle None gradients (for robustness)
            if gradients is not None:
                valid_gradients = []
                valid_vars = []

                for grad, var in zip(gradients, trainable_variables):
                    if grad is not None:
                        valid_gradients.append(grad)
                        valid_vars.append(var)

                if valid_gradients:
                    optimizer.apply_gradients(zip(valid_gradients, valid_vars))
            else:
                print("No gradients calculated in this training step. Skipping update for this step.")

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

        return loss
    
# composer.py
class FunctionGraphComposer:  # Composer class (separate)
    def __init__(self):
        self.graph = FunctionGraph()
        self.input_shape = None

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        return self  # Chaining

    def add(self, node):
        self.graph.add(node)
        return self  # Chaining

    def connect(self, from_node_name, to_node_name):
        from_node = self.graph.nodes[from_node_name]
        to_node = self.graph.nodes[to_node_name]

        if to_node.input_shape is None:  # Build the to_node if it hasn't been built yet.
            if from_node == self.graph.input_node and self.input_shape is not None:  # Correctly handle the first connection when input shape is rank 1
                to_node.build(self.input_shape)
            else:
                to_node.build(from_node.output_shape)  # Build the to_node based on input from from_node

        self.graph.connect(from_node_name, to_node_name)  # Correct: self.graph.connect
        return self  # Chaining

    def set_input(self, node_name):
        self.graph.set_input(node_name)
        return self  # Chaining

    def set_output(self, node_name):
        self.graph.set_output(node_name)
        return self  # Chaining

    def build(self):
        if self.graph.input_node is None:
            raise ValueError("Input node must be set.")

        if self.input_shape is None:
            raise ValueError("Input shape must be set before building the graph.")

        self.graph.input_shape = self.input_shape
        self.graph.input_node.build(self.input_shape)  # Build input node first

        # Correctly build nodes with dependencies:
        built_nodes = {self.graph.input_node.name}  # Keep track of built nodes

        while len(built_nodes) < len(self.graph.nodes):
            for node in self.graph.nodes.values():
                if node.name not in built_nodes:
                    incoming_connections = self.graph.connections.get(node.name, [])
                    ready_to_build = True
                    input_shapes = []

                    if not incoming_connections:  # Handle case where the node is directly connected to the input node.
                        incoming_connections = [k for k, v in self.graph.nodes.items() if v == self.graph.input_node]

                    for conn in incoming_connections:
                        from_node = self.graph.nodes[conn]
                        if from_node.name not in built_nodes:
                            ready_to_build = False
                            break  # Not ready to build if a dependency isn't built

                        input_shapes.append(from_node.output_shape)

                    if ready_to_build:
                        if len(input_shapes) == 1:
                            node.build(input_shapes[0])
                        else:  # Multiple inputs:
                            concatenated_shape = list(input_shapes[0])
                            concatenated_shape[-1] = sum(s[-1] for s in input_shapes)
                            node.build(tuple(concatenated_shape))

                        built_nodes.add(node.name)  # Mark the node as built
                        break  # Important to break out of the inner loop, and go back to the beginning, to make sure the next node is built.
            else:  # If the for loop completes without building a node, it means there is a circular dependency.
                unbuilt_nodes = [node.name for node in self.graph.nodes.values() if node.name not in built_nodes]
                raise ValueError(f"Circular dependency detected. Could not build nodes: {unbuilt_nodes}")

        return self

    def compile(self, optimizer, loss_function):
        self.optimizer = optimizer
        self.loss_function = loss_function
        return self

    def train(self, inputs, targets, epochs=1, verbose=0):
        if not hasattr(self, 'optimizer') or not hasattr(self, 'loss_function'):
            raise ValueError("Graph must be compiled before training.")
        return self.graph.train(inputs, targets, self.optimizer, self.loss_function, epochs, verbose)

    def forward(self, inputs):
        return self.graph.forward(inputs)