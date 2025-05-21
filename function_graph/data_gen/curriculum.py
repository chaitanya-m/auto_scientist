# data_gen/curriculum.py
import numpy as np


class Curriculum:
    def __init__(self, config: dict):
        """
        Manages a curriculum of autoencoder tasks.
        Args:
            config: Configuration dictionary.
        """
        self.config = config

