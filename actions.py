from abc import ABC, abstractmethod

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


# Example usage
if __name__ == "__main__":
    multiply_action = MultiplyAction(2)

    print(multiply_action.execute(10))  # Output: 20

