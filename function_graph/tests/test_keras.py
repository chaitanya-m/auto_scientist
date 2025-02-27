import unittest
import numpy as np
from tensorflow import keras

class TestFrozenLayerBehavior(unittest.TestCase):
    """Validate core behaviors required for compatibility with our graph node storage system."""
    
    def test_gradient_flow_through_frozen_layers(self):
        """Verify gradients propagate through frozen layers to earlier trainable layers."""
        # Model architecture
        inputs = keras.Input(shape=(10,))
        frozen = keras.layers.Dense(32, activation='relu', trainable=False)(inputs)
        outputs = keras.layers.Dense(1)(frozen)
        model = keras.Model(inputs, outputs)
        
        # Training setup
        model.compile(optimizer='adam', loss='mse')
        model.fit(np.random.rand(16, 10), np.random.rand(16, 1), epochs=1, verbose=0)
        
        # Weight comparison
        frozen_weights = model.layers[1].get_weights()
        output_weights = model.layers[2].get_weights()
        
        # Validate frozen layer
        self.assertTrue(  # Check weights and biases separately
            np.allclose(frozen_weights[0], model.layers[1].get_weights()[0]),
            "Frozen layer weights changed unexpectedly"
        )
        self.assertTrue(
            np.allclose(frozen_weights[1], model.layers[1].get_weights()[1]),
            "Frozen layer biases changed unexpectedly"
        )
        
        # Validate trainable layer
        self.assertFalse(
            np.allclose(output_weights[0], np.zeros_like(output_weights[0])),
            "Output layer weights failed to update"
        )

    def test_optimizer_adaptation_with_mixed_trainability(self):
        """Validate Adam optimizer properly handles mixed trainable layers."""
        # Model architecture
        inputs = keras.Input(shape=(15,))
        x = keras.layers.Dense(32)(inputs)
        x = keras.layers.Dense(16, trainable=False)(x)
        outputs = keras.layers.Dense(8)(x)
        model = keras.Model(inputs, outputs)
        
        # Track only weight-bearing layers
        trainable_layers = [
            layer for layer in model.layers 
            if layer.trainable and hasattr(layer, 'kernel')
        ]
        initial_weights = [layer.get_weights()[0].copy() for layer in trainable_layers]
        
        # Training execution
        model.compile(optimizer='adam', loss='mse')
        model.fit(np.random.rand(64, 15), np.random.rand(64, 8), epochs=3, verbose=0)
        
        # Verify weight updates
        for i, layer in enumerate(trainable_layers):
            final_kernel = layer.get_weights()[0]
            initial_kernel = initial_weights[i]
            weight_change = np.mean(np.abs(final_kernel - initial_kernel))
            self.assertGreater(
                weight_change, 1e-4,
                f"Layer {layer.name} weights changed insufficiently (Δ={weight_change:.2e})"
            )

    def test_partial_freezing_with_nested_components(self):
        """Confirm complex freezing patterns behave as expected."""
        # Model construction
        inputs = keras.Input(shape=(12,))
        x = keras.layers.Dense(24)(inputs)
        x = keras.layers.Dense(18, trainable=False)(x)
        x = keras.layers.Dense(12)(x)
        x = keras.layers.Dense(6, trainable=False)(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        
        # InputLayer should never contribute trainable weights
        input_layer = next(l for l in model.layers if isinstance(l, keras.layers.InputLayer))
        self.assertEqual(
            len(input_layer.trainable_weights),
            0,
            "InputLayer should never have trainable weights. "
            f"Found {len(input_layer.trainable_weights)} trainable parameters."
        )
        
        # Verify Dense layer trainability pattern
        dense_layers = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
        expected_trainability = [
            True,   # Dense(24)
            False,  # Dense(18)
            True,   # Dense(12)
            False,  # Dense(6)
            True    # Dense(1)
        ]
        
        actual_trainability = [layer.trainable for layer in dense_layers]
        self.assertEqual(
            actual_trainability,
            expected_trainability,
            "Dense layer trainability pattern mismatch\n"
            f"Expected: {expected_trainability}\n"
            f"Actual:   {actual_trainability}"
        )

if __name__ == "__main__":
    unittest.main()
