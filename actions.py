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
    '''

    def __init__(self, environment, multiplier, cutoff_low=None, cutoff_high=None):
        super().__init__(multiplier, cutoff_low, cutoff_high)
        self.environment = environment

    def execute(self):
        if self.environment:
            print("Old delta:" + str(self.environment.model.delta))
            self.environment.model.delta = super().execute(self.environment.model.delta)
            print("New delta:" + str(self.environment.model.delta))
        else:
            raise ValueError("Environment not set")

    def get_params(self):
        params = super().get_params()
        params["environment_delta"] = self.environment.model.delta if self.environment else None
        return params

# Example usage
if __name__ == "__main__":
    multiply_action = MultiplyAction(2)

    print(multiply_action.execute(10))  # Output: 20

