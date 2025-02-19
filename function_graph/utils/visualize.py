import os
from graphviz import Digraph
from keras import utils
from graph.node import SubGraphNode

def visualize_graph(composer, output_file_base="graph_view", image_dir="graph_images"):
    """
    Visualizes the composed graph as a single image with subgraph images displayed
    next to their corresponding nodes.

    For each node in the composer:
      - A text-only node is drawn with the node's name and type.
      - If the node is a SubGraphNode, an SVG image of its internal Keras model is generated
        (via keras.utils.plot_model), and a separate image node is added to the graph.
    An invisible edge is created from the original subgraph node to its image node,
    and a subgraph cluster is defined to force them onto the same rank.
    
    Each call increments a counter so that output filenames are unique.
    The final composite graph is rendered as an SVG, automatically opened, and its DOT source is printed.
    """
    # Create directory for images if it does not exist.
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # Ensure unique file names by maintaining a counter.
    if not hasattr(visualize_graph, "counter"):
        visualize_graph.counter = 0
    visualize_graph.counter += 1
    output_file = f"{output_file_base}_{visualize_graph.counter}"
    
    dot = Digraph(comment="Neural Net Graph", format="svg")
    
    # Set general graph attributes for layout and font.
    dot.attr(rankdir="LR")
    dot.attr(fontname="Arial", fontsize="10")
    dot.attr('node', fontname="Arial", fontsize="10", shape="box", style="filled", fillcolor="white")
    
    # Dictionary to store names of image nodes for each SubGraphNode.
    subgraph_image_nodes = {}
    
    # Add each node in the composer.
    for node_name, node in composer.nodes.items():
        label = f"{node_name}\\n({node.__class__.__name__})"
        dot.node(node_name, label=label)
        if isinstance(node, SubGraphNode):
            image_filename = os.path.join(image_dir, f"{node_name}_{visualize_graph.counter}.svg")
            try:
                # Generate an SVG image of the subgraph's internal Keras model.
                utils.plot_model(node.model, to_file=image_filename, show_shapes=True, dpi=60)
                # Create a separate node for the image.
                image_node_name = f"{node_name}_img"
                dot.node(image_node_name, label="", image=image_filename, shape="none")
                # Add an invisible edge to force the image node to appear next to the original node.
                dot.edge(node_name, image_node_name, style="dashed", constraint="false")
                subgraph_image_nodes[node_name] = image_node_name
            except Exception as e:
                print(f"Error generating plot for node '{node_name}': {e}")
    
    # Add the edges from the composer.
    for target, connections in composer.connections.items():
        for parent, merge_mode in connections:
            dot.edge(parent, target, label=merge_mode)
    
    # Create subgraph clusters for each SubGraphNode and its image node so that they appear in the same rank.
    for node_name, image_node in subgraph_image_nodes.items():
        with dot.subgraph() as s:
            s.attr(rank="same")
            s.node(node_name)
            s.node(image_node)
    
    # Render the graph, save to file, and open it.
    dot.render(output_file, view=True)
    print(f"Graph visualized and saved to {output_file}.svg")
    
    # Print the DOT source for debugging.
    print("\nGraph DOT source:")
    print(dot.source)


def print_graph_nodes(composer):
    """
    Prints a visual summary of the graph constructed by the composer.
    For each node, it prints:
      - The node name and its type.
      - For InputNodes: the input shape.
      - For SingleNeuron nodes: the activation function and the weights (if the model is built).
      - For SubGraphNodes: a summary of the contained Keras model and the weights of each layer.
    Finally, it prints the connection list.
    """
    print("=== Graph Visualizer ===")
    print("Nodes:")
    for node_name, node in composer.nodes.items():
        print(f"Node: {node_name} ({node.__class__.__name__})")
        # If the node has an input shape attribute (e.g. InputNode), print it.
        if hasattr(node, "input_shape"):
            print(f"  Input shape: {node.input_shape}")
        # For SingleNeuron, print activation and (if available) its weights.
        if node.__class__.__name__ == "SingleNeuron":
            print(f"  Activation: {node.activation}")
            try:
                # Assuming the model has been built, retrieve the corresponding layer.
                layer = composer.keras_model.get_layer(node_name)
                weights = layer.get_weights()
                print(f"  Weights: {weights}")
            except Exception as e:
                print("  Weights not available (model not built yet).")
        # For SubGraphNode, print the internal Keras model summary and each layer's weights.
        if node.__class__.__name__ == "SubGraphNode":
            print("  Contains a Keras model:")
            node.model.summary()
            for layer in node.model.layers:
                print(f"    Layer {layer.name} weights: {layer.get_weights()}")
    print("\nConnections:")
    for target, connections in composer.connections.items():
        print(f"Node '{target}' has incoming connections:")
        for parent, merge_mode in connections:
            print(f"  From: '{parent}', merge mode: {merge_mode}")
    print("=== End of Graph ===")
