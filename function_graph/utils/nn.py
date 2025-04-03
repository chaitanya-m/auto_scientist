# utils/nn.py
from graph.composer import GraphComposer, GraphTransformer
from graph.node import InputNode, SingleNeuron, SubGraphNode
import tensorflow as tf

def train_and_evaluate(model, dataset, train_ratio=0.5, epochs=1, verbose=0):
    """
    Splits 'dataset' into training and testing sets according to train_ratio,
    trains 'model' on the training set, and evaluates on the test set.

    Parameters
    ----------
    model : keras.Model
        The Keras model to train and evaluate.
    dataset : pd.DataFrame
        A DataFrame containing columns like 'feature_0', 'feature_1', ..., 'label', etc.
    train_ratio : float
        Fraction of the dataset to use for training (between 0 and 1).
    epochs : int
        Number of training epochs.
    verbose : int
        Verbosity mode for model.fit and model.predict.

    Returns
    -------
    float
        The accuracy on the test split.
    """
    split_idx = int(len(dataset) * train_ratio)
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]

    train_features = train_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
    train_labels = train_df["label"].to_numpy(dtype=int)

    test_features = test_df[[f"feature_{i}" for i in range(2)]].to_numpy(dtype=float)
    test_labels = test_df["label"].to_numpy(dtype=int)

    model.fit(train_features, train_labels, epochs=epochs, verbose=verbose)
    predictions = model.predict(test_features, verbose=verbose)
    preds = (predictions.flatten() > 0.5).astype(int)
    accuracy = (preds == test_labels).mean()
    return accuracy


def create_minimal_graphmodel(input_shape):
    """
    Builds a minimal valid network using our node and composer framework.
    The network has an InputNode (with the given input_shape) and a SingleNeuron output node with sigmoid activation.
    """
    composer = GraphComposer()
    input_node = InputNode(name="input", input_shape=input_shape)
    # For binary classification, we use a sigmoid on the output.
    output_node = SingleNeuron(name="output", activation="sigmoid")
    composer.add_node(input_node)
    composer.add_node(output_node)
    composer.set_input_node("input")
    composer.set_output_node("output")
    composer.connect("input", "output")
    model = composer.build()
    return composer, model

class AdamWithLRMultiplier(tf.keras.optimizers.Adam):
    def __init__(self, lr_map, *args, **kwargs):
        """
        lr_map: dict mapping substrings in variable names to a gradient multiplier.
                e.g. {'output': 5.0} scales gradients for any variable whose name includes 'output'.
        All other arguments are passed to the Adam initializer.
        """
        super().__init__(*args, **kwargs)
        self.lr_map = lr_map

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        new_grads_and_vars = []
        for grad, var in grads_and_vars:
            print("Variable name: " + var.name)
            if grad is not None:
                multiplier = 1.0
                # First, try checking for an attribute we set on the variable.
                if hasattr(var, "_layer_name"):
                    for key_substring, factor in self.lr_map.items():
                        if key_substring in var._layer_name:
                            multiplier = factor
                            break
                else:
                    # Fallback: check the variable name.
                    for key_substring, factor in self.lr_map.items():
                        if key_substring in var.name:
                            multiplier = factor
                            break
                grad = grad * multiplier
            new_grads_and_vars.append((grad, var))
        return super().apply_gradients(new_grads_and_vars, **kwargs)



class AdamWithPostScaleMultiplier(tf.keras.optimizers.Adam):
    def __init__(self, lr_map, *args, **kwargs):
        """
        lr_map: dict mapping substrings in variable (or layer) names to a post-scale multiplier.
                For example, {'fast': 5.0} means that for any variable whose associated layer's name
                contains 'fast', the final update computed by Adam will be multiplied by 5.
        All other arguments are passed to the Adam initializer.
        """
        super().__init__(*args, **kwargs)
        self.lr_map = lr_map

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Determine the multiplier for this variable.
        multiplier = 1.0
        # First, try using our annotation (set via annotate_model_variables).
        if hasattr(var, "_layer_name"):
            for key_substring, factor in self.lr_map.items():
                if key_substring in var._layer_name:
                    multiplier = factor
                    break
        else:
            for key_substring, factor in self.lr_map.items():
                if key_substring in var.name:
                    multiplier = factor
                    break

        # Get the current optimizer state for this variable.
        var_dtype = var.dtype.base_dtype
        coefficients = self._prepare_local(var_dtype)
        # Get slots for m and v (the first and second moment estimates).
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Increment iteration counter.
        local_step = tf.cast(self.iterations + 1, var_dtype)

        beta1_power = tf.pow(self._get_hyper("beta_1", var_dtype), local_step)
        beta2_power = tf.pow(self._get_hyper("beta_2", var_dtype), local_step)
        lr_t = self._decayed_lr(var_dtype)

        # Update biased first moment estimate.
        m_t = m.assign(self._get_hyper("beta_1", var_dtype) * m + (1 - self._get_hyper("beta_1", var_dtype)) * grad,
                       use_locking=self._use_locking)
        # Update biased second raw moment estimate.
        v_t = v.assign(self._get_hyper("beta_2", var_dtype) * v + (1 - self._get_hyper("beta_2", var_dtype)) * tf.square(grad),
                       use_locking=self._use_locking)
        # Compute bias-corrected first moment estimate.
        m_hat = m_t / (1 - beta1_power)
        # Compute bias-corrected second moment estimate.
        v_hat = v_t / (1 - beta2_power)
        # Compute the standard Adam update.
        update = lr_t * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        # Post-scale the update by our multiplier.
        update = update * multiplier

        var_update = var.assign_sub(update, use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # For sparse updates, a similar approach is needed. Here we provide a simple implementation
        # that follows the same idea: compute the update and post-scale it.
        multiplier = 1.0
        if hasattr(var, "_layer_name"):
            for key_substring, factor in self.lr_map.items():
                if key_substring in var._layer_name:
                    multiplier = factor
                    break
        else:
            for key_substring, factor in self.lr_map.items():
                if key_substring in var.name:
                    multiplier = factor
                    break

        var_dtype = var.dtype.base_dtype
        coefficients = self._prepare_local(var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta1_power = tf.pow(self._get_hyper("beta_1", var_dtype), local_step)
        beta2_power = tf.pow(self._get_hyper("beta_2", var_dtype), local_step)
        lr_t = self._decayed_lr(var_dtype)

        # Sparse update for m and v.
        m_scaled_g_values = grad * (1 - self._get_hyper("beta_1", var_dtype))
        m_t = m.assign(self._get_hyper("beta_1", var_dtype) * m, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        v_scaled_g_values = (grad * grad) * (1 - self._get_hyper("beta_2", var_dtype))
        v_t = v.assign(self._get_hyper("beta_2", var_dtype) * v, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_hat = m_t / (1 - beta1_power)
        v_hat = v_t / (1 - beta2_power)
        update = lr_t * m_hat / (tf.sqrt(v_hat) + self.epsilon)
        update = update * multiplier
        var_update = self._resource_scatter_add(var, indices, -update)
        return tf.group(*[var_update, m_t, v_t])


class SGDWithPostScaleMultiplier(tf.keras.optimizers.SGD):
    def __init__(self, lr_map, *args, **kwargs):
        """
        lr_map: dict mapping substrings (in layer names) to a multiplier.
        For example, {'fast': 5.0} means that for any variable whose associated layer's name contains 'fast',
        the final update will be multiplied by 5.
        """
        super().__init__(*args, **kwargs)
        self.lr_map = lr_map

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        new_grads_and_vars = []
        for grad, var in grads_and_vars:
            multiplier = 1.0
            if hasattr(var, "_layer_name"):
                for key_substring, factor in self.lr_map.items():
                    if key_substring in var._layer_name:
                        multiplier = factor
                        break
            else:
                for key_substring, factor in self.lr_map.items():
                    if key_substring in var.name:
                        multiplier = factor
                        break
            # Multiply the gradient by the multiplier.
            if grad is not None:
                grad = grad * multiplier
            new_grads_and_vars.append((grad, var))
        return super().apply_gradients(new_grads_and_vars, **kwargs)
