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
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def execute(self, value):
        return value * self.multiplier

    def get_params(self):
        return {"multiplier": self.multiplier}


# Example usage
if __name__ == "__main__":
    multiply_action = MultiplyAction(2)

    print(multiply_action.execute(10))  # Output: 20

