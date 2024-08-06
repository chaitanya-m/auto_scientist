from abc import ABC, abstractmethod
# Actions have access to the environment and can change the environment's state

# Base Action Class
class BaseAction(ABC):
    @abstractmethod
    def execute(self, value):
        pass

    @abstractmethod
    def get_params(self):
        pass

    # Now add a method that adds an env
    def set_env(self, env):
        '''
        Should be set by the environment when the action is added to the environment
        '''
        self.env = env

# Algorithm State Modification Action Class
class ModifyAlgorithmStateAction(BaseAction):
    '''
    Flips the values at specific indices of the algorithm state vector

    Args:
        indices (list of int): The indices to modify in the state vector
    '''

    def __init__(self, indices):
        if not all(isinstance(index, int) for index in indices):
            raise ValueError("All indices must be integers.")
        self.indices = indices
        self.env = None
        self.state = None

    def execute(self, _):
        if self.state is not None:
            for index in self.indices:
                current_value = self.state[index]
                new_value = 1 - current_value  # Flip the value
                self.state.set_item(index, new_value)
        else:
            raise ValueError("State is not set for this action.")

    def get_params(self):
        return {"indices": self.indices}


# Multiply Action Class
class MultiplyAction(BaseAction):
    '''
    Multiplies the input value by a multiplier

    Args:
        multiplier (float): The value to multiply the input by
        cutoff_low (float): The minimum value allowed
        cutoff_high (float): The maximum value allowed

    Returns:
        float: The result of the multiplication

    Note: If cutoff_low and cutoff_high are provided, the result will be clipped to the range [cutoff_low, cutoff_high] 
    '''

    def __init__(self, multiplier, cutoff_low = None, cutoff_high = None):
        self.multiplier = multiplier
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.env = None

    def execute(self, value):
        result = value * self.multiplier
        if self.cutoff_low is not None and self.cutoff_high is not None:
            if result > self.cutoff_high:
                return self.cutoff_high
            elif result < self.cutoff_low:
                return self.cutoff_low
        return result

    def get_params(self):
        return {"multiplier": self.multiplier}

# MultiplyDeltaAction Class
class MultiplyDeltaAction(MultiplyAction):
    '''
    Multiplies the environment's delta_hard by a multiplier
    Note: env must be passed in when the action is created. Because of circular dependency, the action is not initialised with env.
    '''

    def __init__(self, multiplier, cutoff_low=None, cutoff_high=None):
        super().__init__(multiplier, cutoff_low, cutoff_high)

    def execute(self):
        if self.env:
            self.env.model.delta = super().execute(self.env.model.delta)
        else:
            raise ValueError("Environment not set for action. Cannot execute action.")

    def get_params(self):
        params = super().get_params()
        params["env_delta"] = self.env.model.delta if self.env else None
        return params


# MethodSwitchAction Class
class MethodSwitchAction(BaseAction):
    def __init__(self, method, current_alternative, other_alternative):
        '''
        None: Environment is not set
        As env may not yet be set when the action is created, it should be set later by the environment using set_env
        The attribute named method will be replaced by current_alternative or other_alternative
        '''
        self.env = None 
        self.method = method
        self.current_alternative = current_alternative
        self.other_alternative = other_alternative

    def execute(self):
        if self.env is None or self.env.model is None:
            raise ValueError("Environment or model is not set.")
        
        # Swap the methods
        if getattr(self.env.model, self.method) == self.current_alternative:
            setattr(self.env.model, self.method, self.other_alternative)
            self.current_alternative, self.other_alternative = self.other_alternative, self.current_alternative
        else:
            setattr(self.env.model, self.method, self.current_alternative)
            self.current_alternative, self.other_alternative = self.other_alternative, self.current_alternative

    def get_params(self):
        return {
            "action": "method_switch",
            "method": self.method,
            "current_alternative": self.current_alternative,
            "other_alternative": self.other_alternative
        }


class CutEFDTStrategySwitchAction(MethodSwitchAction):
    def __init__(self, method, current_alternative, other_alternative):
        super().__init__(method, current_alternative, other_alternative)

    def execute(self):
        super().execute()

    def get_params(self):
        params = super().get_params()
        params["strategy"] = "CutEFDT_strategy_enable_disable"
        return params


class SetMethodAction(BaseAction):
    def __init__(self, methods_to_update):
        '''
        None: Environment is not set
        As env may not yet be set when the action is created, it should be set later by the environment using set_env
        The attribute named method will be set to method_to_use
        '''
        self.env = None
        self.methods_to_update = methods_to_update

    def execute(self):
        if self.env is None or self.env.model is None:
            raise ValueError("Environment or model is not set.")
        else:
            for method, method_to_use in self.methods_to_update.items():
                new_method = getattr(self.env.model, method_to_use)
                setattr(self.env.model, method, new_method)

    def get_params(self):
        '''
        Returns the parameters of the action
        '''
        return {
            "action": "method_update",
            "methods_to_update": self.methods_to_update,
        }


class SetEFDTStrategyAction(SetMethodAction):
    def __init__(self, strategies):
        super().__init__(strategies)

    def execute(self):
        super().execute()

    def get_params(self):
        params = super().get_params()
        params["strategy"] = "EFDT_strategy_enable_disable"
        return params


# Example usage
if __name__ == "__main__":
    multiply_action = MultiplyAction(2)

    print(multiply_action.execute(10))  # Output: 20

