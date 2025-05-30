import numpy as np
import pandas as pd
from itertools import product
from abc import ABC, abstractmethod

class SyntheticDataGenerator(ABC):
    @abstractmethod
    def generate_data(self, num_instances: int):
        pass

class DataSchema(SyntheticDataGenerator):
    def __init__(self, num_features, num_categories, num_classes, relationship_mapping, rng=None):
        self.num_features = num_features
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.relationship_mapping = relationship_mapping
        # Store an RNG internally; if none provided, create one without a fixed seed.
        self.rng = rng if rng is not None else np.random.default_rng()

    def assign_class(self, feature_vector):
        return self.relationship_mapping[tuple(feature_vector)]

    def generate_dataset(self, num_instances):
        rng = self.rng
        # 1. Get class distribution from schema.
        class_labels = list(self.relationship_mapping.values())
        unique_classes, class_counts = np.unique(class_labels, return_counts=True)
        class_probs = class_counts / len(class_labels)
        # 2. Generate class labels.
        labels = rng.choice(unique_classes, size=num_instances, p=class_probs)
        # 3. For each label, sample one feature vector.
        features = []
        for label in labels:
            possible_fvs = [fv for fv, cl in self.relationship_mapping.items() if cl == label]
            fv = possible_fvs[rng.integers(0, len(possible_fvs))]
            features.append(fv)
        data = np.concatenate((features, labels.reshape(-1, 1)), axis=1)
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(self.num_features)] + ["label"])
        return df

    # For compatibility with the interface, alias generate_dataset as generate_data.
    def generate_data(self, num_instances: int):
        return self.generate_dataset(num_instances)
    
    def print_schema(self):
        print(f"Number of Features: {self.num_features}")
        print(f"Number of Categories per Feature: {self.num_categories}")
        print(f"Number of Classes: {self.num_classes}")
        print("Relationship Mapping (limited to first 10 entries):")
        for i, (fv, cl) in enumerate(self.relationship_mapping.items()):
            if i < 10:
                print(f"  {fv}: {cl}")
            else:
                break


class DataSchemaFactory:
    def create_schema(self, num_features, num_categories, num_classes, flatness=1.0, random_seed=None):
        """
        Creates a DataSchema object based on the specified parameters.
        
        Parameters:
        -----------
        num_features : int
            The number of features in each data instance.
        num_categories : int
            The number of possible categories for each feature.
        num_classes : int
            The total number of classes to choose from.
        flatness : float, optional (default=1.0)
            Controls the concentration of the Dirichlet distribution used to generate
            class probabilities. Higher values lead to more uniform distributions.
        random_seed : int, optional
            The seed for the random number generator. Using the same seed across multiple
            calls ensures that the same random numbers (and therefore the same schema) are produced.
        
        Returns:
        --------
        DataSchema
            A DataSchema object containing:
              - num_features, num_categories, and num_classes,
              - a relationship mapping that assigns a class label to each possible feature vector,
              - a random number generator (rng) initialized with the given seed.
        
        Behavior with the same seed:
        -----------------------------
        When create_schema is called multiple times with the same random_seed, the underlying
        np.random.default_rng(random_seed) creates a new random generator that is initialized to 
        the same state. Therefore, the Dirichlet distribution (used to generate class_distribution)
        and the subsequent calls to rng.choice() will produce identical outputs. As a result, the 
        'relationship_mapping' generated will be the same in every call with the same seed.
        This ensures reproducibility in experiments: the schema is deterministic when the seed is fixed.
        """
        # Initialize a new random number generator with the provided seed.
        rng = np.random.default_rng(random_seed)
        
        # Generate a class probability distribution using a Dirichlet distribution.
        # The flatness parameter controls the concentration: a flatness of 1.0 leads to a uniform-like distribution.
        class_distribution = rng.dirichlet(np.ones(num_classes) * flatness)
        
        # Initialize an empty mapping from feature vectors to class labels.
        relationship_mapping = {}
        
        # For each possible feature vector (using Cartesian product of categories for each feature),
        # randomly assign a class label based on the generated class distribution.
        for feature_vector in product(range(num_categories), repeat=num_features):
            # rng.choice with the same seed and same probability distribution will always choose the same class label.
            class_label = rng.choice(num_classes, p=class_distribution)
            relationship_mapping[feature_vector] = class_label
        
        # Create and return a DataSchema object with the generated parameters and relationship mapping.
        return DataSchema(num_features, num_categories, num_classes, relationship_mapping, rng=rng)

    def generate_schemas_and_datasets(self, schema_types, num_schemas_per_type, random_seed_start):
        for schema_type in schema_types:
            for i in range(num_schemas_per_type):
                random_seed = random_seed_start + i
                schema = self.create_schema(**schema_types[schema_type], random_seed=random_seed)
                df = schema.generate_data(num_instances=100)
                yield schema_type, schema, df
            random_seed_start += num_schemas_per_type
