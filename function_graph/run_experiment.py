# Define schema types
# Create a DataSchemaFactory
# Set up a generator to produce schemas and datasets for experimentation
# Take datasets from generator and run experiments

import function_graph.data_gen.categorical_classification as categorical_classification

# Create an instance of the DataSchemaFactory to generate schemas
factory = categorical_classification.DataSchemaFactory()

# Define the types of schemas to generate, with their parameters
# Each schema type is a dictionary with keys:
#   - "num_features": Number of features in the schema
#   - "num_categories": Number of categories per feature
#   - "num_classes": Number of output classes
#   - "flatness": Controls the class distribution (higher values = more uniform)
schema_types = {
    "type1": {"num_features": 2, "num_categories": 2, "num_classes": 2, "flatness": 10},
    "type2": {"num_features": 3, "num_categories": 3, "num_classes": 3, "flatness": 5},
}

# Specify the number of schemas to generate for each type
num_schemas_per_type = 10

# Set the starting value for the random seed used in schema generation
random_seed_start = 0

# Create a generator object that will yield schemas and datasets one by one
# The generator takes the schema types, number of schemas per type, and starting random seed as input
dataset_generator = factory.generate_schemas_and_datasets(schema_types, num_schemas_per_type, random_seed_start)


for i, (schema_type, schema, dataset) in enumerate(dataset_generator):

    print(f"\nProcessing schema type: {schema_type}, Iteration: {i}")
    print(f"Schema details:")
    schema.print_schema()

    print(f"Dataset head:")
    print(dataset.head())
    