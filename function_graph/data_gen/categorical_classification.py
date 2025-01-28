import numpy as np
import pandas as pd
from itertools import product

class DataSchema:
    def __init__(self, num_features, num_categories, num_classes, relationship_mapping):
        self.num_features = num_features
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.relationship_mapping = relationship_mapping

    def assign_class(self, feature_vector):
        return self.relationship_mapping[tuple(feature_vector)]

    def generate_dataset(self, num_instances, random_seed=None):
        rng = np.random.default_rng(random_seed)

        # 1. Get class distribution from schema
        class_labels = list(self.relationship_mapping.values())
        unique_classes, class_counts = np.unique(class_labels, return_counts=True)
        class_probs = class_counts / len(class_labels)  # P(Y)

        # 2. Generate class labels based on P(Y)
        labels = rng.choice(unique_classes, size=num_instances, p=class_probs)

        # 3. Sample feature vectors for each class
        features = []
        for label in labels:
            # Find feature vectors corresponding to this class
            possible_fvs = [fv for fv, cl in self.relationship_mapping.items() if cl == label]
            # Sample a feature vector uniformly from possible options
            fv = possible_fvs[rng.integers(0, len(possible_fvs))]  
            features.append(fv)

        # Combine features and labels into a DataFrame
        data = np.concatenate((features, labels.reshape(-1, 1)), axis=1)  # Reshape labels to a column vector
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(self.num_features)] + ["label"])

        return df
    
    def print_schema(self):
        """Prints the schema information."""
        print(f"Number of Features: {self.num_features}")
        print(f"Number of Categories per Feature: {self.num_categories}")
        print(f"Number of Classes: {self.num_classes}")
        print("Relationship Mapping (limited to first 10 entries):")
        for i, (feature_vector, class_label) in enumerate(self.relationship_mapping.items()):
            if i < 10:
                print(f"  Feature Vector: {feature_vector}, Class Label: {class_label}")
            else:
                break


class DataSchemaFactory:
    def create_schema(self, num_features, num_categories, num_classes, 
                      flatness=1.0, random_seed=None):
        
        rng = np.random.default_rng(random_seed)

        class_distribution = rng.dirichlet(np.ones(num_classes) * flatness)

        relationship_mapping = {}
        for feature_vector in product(range(num_categories), repeat=num_features):
            class_label = rng.choice(num_classes, p=class_distribution)
            relationship_mapping[feature_vector] = class_label

        return DataSchema(num_features, num_categories, num_classes, relationship_mapping)

    def generate_schemas_and_datasets(self, schema_types, num_schemas_per_type, random_seed_start):
        """Generates schemas and datasets one by one using a generator."""

        for schema_type in schema_types:
            for i in range(num_schemas_per_type):
                random_seed = random_seed_start + i  
                schema = self.create_schema(**schema_types[schema_type], random_seed=random_seed) 

                df = schema.generate_dataset(num_instances=100, random_seed=123)  
                
                # Yield the schema and dataset as a tuple
                yield schema_type, schema, df 

            random_seed_start += num_schemas_per_type

