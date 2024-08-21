import numpy as np

from river import tree
from river.datasets.synth import RandomRBF, RandomTree, Sine, Hyperplane, Waveform, SEA, STAGGER, Friedman, Mv, Planes2D
from river.datasets import ImageSegments
from collections import OrderedDict
from actions import MultiplyDeltaAction, SetEFDTStrategyAction, SetMethodAction
import copy

BINS = [1, 2, 3, 4]
NUM_ACCURACY_CHANGE_BINS = len(BINS) * len(BINS)


# Define an AlgorithmState class to store the state of the algorithm
# Algorithms are vectors with each element representing a binary choice, 0 or 1
# What we want to see is whether regardless of the state vector initialization, 
# whether we will have the same convergence of state-action values, with EFDT as the optimal choice, matching also the best prequential accuracy


class AlgorithmState:
    def __init__(self, vector):
        if not all(v in [0, 1] for v in vector):
            raise ValueError("All elements of the vector must be 0 or 1.")
        self.vector = vector

    def __repr__(self):
        return f"State({self.vector})"

    def __eq__(self, other):
        if not isinstance(other, AlgorithmState):
            return False
        return self.vector == other.vector

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, index):
        return self.vector[index]

    def set_item(self, index, value):
        if value not in [0, 1]:
            raise ValueError("Value must be 0 or 1.")
        self.vector[index] = value


class UpdatableHoeffdingTreeClassifier(tree.HoeffdingTreeClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta

class UpdatableEFDTClassifier(tree.ExtremelyFastDecisionTreeClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)

    def update_delta(self, new_delta):
        self.delta = new_delta

class CutEFDTClassifier(UpdatableEFDTClassifier):
    def __init__(self, delta):
        super().__init__(delta=delta)
        self.original_reevaluate_best_split = super()._reevaluate_best_split
        self.original_attempt_to_split = super()._attempt_to_split

        self._reevaluate_best_split = self.reevaluate_best_split_removed
        self._attempt_to_split = self.attempt_to_split_removed

    def update_delta(self, new_delta):
        self.delta = new_delta

    # Now remove the split update mechanism and compare with second best instead of current best
    # Let's begin by removing the update mechanism
    # The superclass tree.ExtremelyFastDecisionTreeClassifier has a method called    def _reevaluate_best_split(self, node, parent, branch_index, **kwargs):
    # It reevaluates the best split for the node.
    # We can override this method to remove the update mechanism
    # For the RL agent, the action is to choose whether to use the overriden update mechanism or the original one... strategy superposition (rather than alternatives)

    def reevaluate_best_split_removed(self, node, parent, branch_index, **kwargs):
        ''' 
            Overridden from superclass(EFDT) to do nothing, to not reevaluate and update splits
            This method is called when a split is reevaluated
            We can override this method to remove the update mechanism
            This method should now do nothing
            If the split revision is to be used, the method should be called from the superclass
        '''
        pass

    # # Now, we need to compare with the second best split instead of the best split
    # # In order to do this, we will use the def _attempt_to_split(self, node, parent, branch_index, **kwargs) from EFDT's superclass, HoeffdingTreeClassifier

    def attempt_to_split_removed(self, node, parent, branch_index, **kwargs):
        ''' 
            Override EFDT split Call HoeffdingTreeClassifier's _attempt_to_split method, in order to compare only with the second best split
            HoeffdingTreeClassifier is cutEFDT's superclass EFDT's superclass
        '''

        # Explicitly call the great-grandparent class method, because using Python's super in series didn't seem to work (or I had the number of super's wrong?)
        tree.HoeffdingTreeClassifier._attempt_to_split(self, node, parent, branch_index, **kwargs)


# Create a dictionary mapping class names to class objects
model_classes = {
    'UpdatableHoeffdingTreeClassifier': UpdatableHoeffdingTreeClassifier,
    'UpdatableEFDTClassifier': UpdatableEFDTClassifier,
    'CutEFDTClassifier': CutEFDTClassifier
}

# Create a dictionary mapping class names to their respective action space

action_spaces = {
    UpdatableHoeffdingTreeClassifier: [MultiplyDeltaAction(1/100, 1e-10, 1), MultiplyDeltaAction(1/10, 1e-10, 1), MultiplyDeltaAction(1, 1e-10, 1), MultiplyDeltaAction(10, 1e-10, 1), MultiplyDeltaAction(100, 1e-10, 1)],
    UpdatableEFDTClassifier: [MultiplyDeltaAction(1/100, 1e-10, 1), MultiplyDeltaAction(1/10, 1e-10, 1), MultiplyDeltaAction(1, 1e-10, 1), MultiplyDeltaAction(10, 1e-10, 1), MultiplyDeltaAction(100, 1e-10, 1)],
    # CutEFDTClassifier: [MultiplyDeltaAction(1, 1e-10, 1),
    #                     MultiplyDeltaAction(1, 1e-10, 1),
    #                     MultiplyDeltaAction(1, 1e-10, 1),
    #                     MultiplyDeltaAction(1, 1e-10, 1),
    #                     MultiplyDeltaAction(1, 1e-10, 1), 
    #                     MultiplyDeltaAction(1/100, 1e-10, 1), 
    #                     MultiplyDeltaAction(100, 1e-10, 1),
    #                     SetEFDTStrategyAction({"_reevaluate_best_split": "original_reevaluate_best_split", "_attempt_to_split": "original_attempt_to_split"}),
    #                     SetEFDTStrategyAction({"_reevaluate_best_split": "reevaluate_best_split_removed", "_attempt_to_split": "attempt_to_split_removed"})
    #                     ]
}

# Design space maps the variables to their respective action spaces
# _reevaluate_best_split can have two values: "original_reevaluate_best_split" and "reevaluate_best_split_removed"
# _attempt_to_split can have two values: "original_attempt_to_split" and "attempt_to_split_removed"
# binary_design_spaces  is a dictionary mapping class names to their respective design spaces
# All design choices should be binary




class Environment:
    '''
    Note: Each Environment instance is associated with a single model and a single baseline model and a single stream factory and a single stream and a single set of actions
    '''

    def __init__(self, state, actions, binary_design_space, model, model_baseline, stream_factory, num_samples_per_epoch, num_epochs_per_episode):

        self.state = state
        self.binary_design_space = binary_design_space
        self.prev_state_index = -1 # Initialize the previous state index to -1

        self.model = model
        self.prev_model = None
        self.model_baseline = model_baseline

        self.apply_design_elements(self.binary_design_space, self.state, SetMethodAction)

        self.current_episode = 0
        self.stream_factory = stream_factory
        self.stream = stream_factory.create(self.current_episode)

        self.accuracy_change_bin = None  # Initialize accuracy change context - 0 indicates no change in accuracy

        self.actions = actions
        for action in self.actions:
            action.set_env(self)

        self.current_epoch = 0
        self.num_epochs = num_epochs_per_episode
        self.num_samples_per_epoch = num_samples_per_epoch
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []

        self.cumulative_accuracy = 0.0
        self.cumulative_baseline_accuracy = 0.0

    def reset(self):
        # Reset the environment
        self.current_episode += 1
        self.last_accuracy = None
        self.last_5_epoch_accuracies = []
        self.current_epoch = 0
        self.accuracy_change_bin = None # Initialize context - accuracy change bin - 0 indicates no change in accuracy
        self.stream = self.stream_factory.create(seed=self.current_episode) # A new seed for the stream
        self.prev_model = None

        self.cumulative_accuracy = 0.0
        self.cumulative_baseline_accuracy = 0.0

        self.state = [np.random.choice([0, 1]) for _ in range(len(self.state))] # Randomly initialize the state vector
        # No matter which algorithm state we start in, we want to see if the agent will converge to the near-optimal state-action 
        # values for EFDT (or some other unknown algorithm in the design space)

        self.prev_state_index = -1 # Initialize the previous state index to -1

        self.apply_design_elements(self.binary_design_space, self.state, SetMethodAction) # Updates the algorithm

        # Return the state index
        return self.index_state_vector(self.state)

    def step(self, action_index):

        # if self.current_epoch == 0:
        #      # first epoch
        #      # its likely that a more eager to split tree will do far better here if noise is low
        #      # We could associate this with an accuracy_change_bin and use that context to determine the next action
        #      # If we pass, we are effectively using this as a burn-in epoch
        #     pass # Burn-in epoch
        # else:

        if self.prev_model is None: # No state changes have occurred yet, so we can use the baseline model as comparison
            self.prev_model = self.model_baseline

        elif action_index < len(self.actions) - 1 : # If the action is not the last (no op) action, there will be a state (algorithm) change
            # Use the current algorithm's model as comparison for the next algorithm's model
            self.prev_state_index = self.index_state_vector(self.state)
            self.prev_model = copy.deepcopy(self.model)
        
        action = self.actions[action_index] # Action that determines updating algorithm bit vector

        print(f"Action Index:  {action_index} State: {self.state}")
        action.execute()
        print(f"Updated State: {self.state}")
        # State vector has been updated by the action



        self.apply_design_elements(self.binary_design_space, self.state, SetMethodAction) # Updated algorithm bit vector applied, algorithm updated
        state_index = self.index_state_vector(self.state)

        # Run one epoch of the experiment
        accuracy, accuracy_prev_model, baseline_epoch_prequential_accuracy = self.run_one_epoch()

        # Increment the epoch counter
        self.current_epoch += 1

        # Update cumulative prequential accuracy and cumulative baseline prequential accuracy
        self.cumulative_accuracy += accuracy
        self.cumulative_baseline_accuracy += baseline_epoch_prequential_accuracy

        # Compute accuracy change of current epoch from the last epoch
        epoch_accuracy_change = 0
        if self.last_accuracy is not None:
            epoch_accuracy_change = ((accuracy - self.last_accuracy) / self.last_accuracy)

        # Update the last accuracy
        self.last_accuracy = accuracy

        # Update the list of last 5 epoch accuracies
        self.last_5_epoch_accuracies.append(accuracy)
        if len(self.last_5_epoch_accuracies) > 5:
            self.last_5_epoch_accuracies.pop(0)

        # Compute accuracy change of current epoch over the average of last 5 epochs
        epoch_5_accuracy_change = 0
        if len(self.last_5_epoch_accuracies) == 5:
            #epoch_5_accuracy_change = ((accuracy - self.last_5_epoch_accuracies[0]) / self.last_5_epoch_accuracies[0]) * 100
            epoch_5_accuracy_change = ((accuracy - np.mean(self.last_5_epoch_accuracies)) / np.mean(self.last_5_epoch_accuracies))

        # Bin the accuracy changes
        epoch_accuracy_change_bin = self.bin_accuracy_change(epoch_accuracy_change)
        epoch_5_accuracy_change_bin = self.bin_accuracy_change(epoch_5_accuracy_change)

        # Update the accuracy change bin
        self.accuracy_change_bin = self.index_accuracy_change_bin(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin)

        # Signal if the episode is done
        done = self.current_epoch == self.num_epochs


        # If we trivially calculate the reward as the difference between the prequential accuracy of the model obtained from reinforcement learning...
        # that unduly rewards past performance. The reward should only take into account improvement over and above the historical improvement.

        # But if we don't reward say a 2% jump on 70% for RL vs a 4% jump on 50% for baseline, we are not rewarding the RL model for its improving performance

        # So we need a formula that always rewards advantage over the baseline, but also rewards improvement over the past performance

        # It is impossible to have a "perfect" reward

        # Eventually we'll have to experiment with evolving the reward (or with different reward functions) to optimise return

        # We should be able to use the accuracy change bin context to guess the state of learning (early in the stream or not, if stationary) and adjust 
        # the reward function accordingly

        if (self.current_epoch == 1):
            reward = 0 # burn-in period - learning has just started. Disable RL as well for first epoch
        else:
            #reward = (accuracy - self.cumulative_accuracy/self.current_epoch) + # reward for improvement over past performance 
            #reward = (accuracy - baseline_epoch_prequential_accuracy)/(self.current_epoch) # linearly decayed reward for improvement over baseline
            reward = 100.0 * (accuracy - accuracy_prev_model) # reward is difference in epoch accuracy between current model and previous model
            # reward = 1 if reward > 0 else -1
            print(f"Accuracy: {accuracy} Prev Accuracy: {accuracy_prev_model}  State Sequence: {self.prev_state_index} {state_index} Action: {action_index} Reward: {reward}")


        return state_index, reward, done

    def apply_design_elements(self, binary_design_space, state, set_method_action_class):
        """
        Applies the design elements based on the provided state vector.

        Args:
            binary_design_space (dict): A dictionary where keys are design element names and values are lists of functions 
                                        or methods corresponding to the possible states (0 or 1).
            state (list): A list representing the current state of each design element, where 0 signifies turning off 
                        a design element and 1 signifies turning it on.
            set_method_action_class (class): The class to be used for setting the design elements. This class should have 
                                            an execute method that applies the design elements.
        """
        design_dict = {}
        state_element = 0

        # For each design element, assign the corresponding function as given by the state vector
        for design_element, design_values in binary_design_space.items():
            # The state vector determines which function to assign to the design element
            design_dict[design_element] = design_values[state[state_element]]
            state_element += 1

        # Create an instance of the action class with the design dictionary and execute it
        write_state = set_method_action_class(design_dict)
        write_state.set_env(self)
        write_state.execute()

        # The updated algorithm (state) has been written to the environment

    @staticmethod
    def index_state_vector(binary_vector):
        """
        Converts a binary vector to an index.

        Args:
            binary_vector (list of int): A list representing a binary vector, where each element is 0 or 1.

        Returns:
            int: The index corresponding to the binary vector.
        """
        index = 0
        length = len(binary_vector)
        
        for i in range(length):
            index += binary_vector[i] * (2 ** (length - i - 1))
        
        return index

    @staticmethod
    def bin_accuracy_change(accuracy_change):
        if accuracy_change <= (5.0/100.0):
            return BINS[0]
        elif accuracy_change <= (20.0/100.0):
            return BINS[1]
        elif accuracy_change <= (50.0/100.0):
            return BINS[2]
        else:
            return BINS[3]

    @staticmethod
    def index_accuracy_change_bin(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin):
        # We take epoch_accuracy_change_bin and epoch_5_accuracy_change_bin and return an accuracy change bin index in the range [0, 24]
        # The index is calculated as below:
        accuracy_change_bin_index = (epoch_accuracy_change_bin - 1) * len(BINS) + (epoch_5_accuracy_change_bin - 1) # 0, 5, 10, 15, 20 correspond to increasingly large 
        return accuracy_change_bin_index

    def action_reintroduce_comparison_with_other_splits(self):
        # If the model is CutEFDTClassifier, reintroduce the comparison with other splits
        if isinstance(self.model, CutEFDTClassifier):
            # Call the _attempt_to_split method from the parent class instead of the current class, which uses the great-grandparent class' method
            # The parent class is UpdatableEFDTClassifier and the great-grandparent class is HoeffdingTreeClassifier
            tree.UpdatableEFDTClassifier._attempt_to_split

    def run_one_epoch(self):
        '''
        Run one epoch of the experiment for both the model and the baseline model (without reinforcement learning) and return the prequential accuracy and reward

        The reward is calculated as the difference between the prequential accuracy of the model and the prequential accuracy of the baseline model.

        The prequential accuracy is calculated as the number of correctly classified samples divided by the total number of samples in the epoch.
        
        Returns:
        - epoch_prequential_accuracy (float): The prequential accuracy for this epoch
        - baseline_epoch_prequential_accuracy(float): Prequential accuracy without agent

        '''

        # Initialize the total accuracy for this epoch
        total_correctly_classified = 0
        total_correctly_classified_prev_model = 0
        total_baseline_correctly_classified = 0
        total_samples = 0

        # Iterate over the data in the stream
        for x, y in self.stream.take(self.num_samples_per_epoch):

            # Predict the output for the current input
            prediction = self.model.predict_one(x)
            prev_model_prediction = self.prev_model.predict_one(x)
            baseline_prediction = self.model_baseline.predict_one(x)

            # Learn from the current input-output pair
            self.model.learn_one(x, y)
            self.prev_model.learn_one(x, y)
            self.model_baseline.learn_one(x, y)


            # Calculate the accuracy of the prediction against the actual output
            is_correctly_classified = self.correctly_classified(prediction, y)
            is_correctly_classified_prev_model = self.correctly_classified(prev_model_prediction, y)
            is_baseline_correctly_classified = self.correctly_classified(baseline_prediction, y)

            # Add the correctly classified to the total correctly classified
            total_correctly_classified += is_correctly_classified
            total_correctly_classified_prev_model += is_correctly_classified_prev_model
            total_baseline_correctly_classified += is_baseline_correctly_classified

            # Increment the total number of samples
            total_samples += 1

        # Calculate the prequential accuracy for this epoch
        epoch_prequential_accuracy = self.average_classification_accuracy(total_correctly_classified, total_samples)
        epoch_prequential_accuracy_prev_model = self.average_classification_accuracy(total_correctly_classified_prev_model, total_samples)
        baseline_epoch_prequential_accuracy = self.average_classification_accuracy(total_baseline_correctly_classified, total_samples)

        # Return the prequential accuracy and baseline_epoch_prequential_accuracy
        return epoch_prequential_accuracy, epoch_prequential_accuracy_prev_model, baseline_epoch_prequential_accuracy

    @staticmethod
    def average_classification_accuracy(correct_predictions, total_predictions):
        return correct_predictions / total_predictions if total_predictions else 0

    @staticmethod
    def correctly_classified(prediction, actual):
        # Check if the prediction is correct
        is_correct = prediction == actual

        # Return 1 if the prediction is correct, 0 otherwise
        return 1 if is_correct else 0


class StreamFactory:
    def __init__(self, stream_type, preinitialized_params):
        self.stream_type = stream_type
        self.preinitialized_params = preinitialized_params

    def create(self, seed):
        if self.stream_type == 'RandomTree':
            return RandomTree(seed_tree=seed, seed_sample=seed, **self.preinitialized_params)
        elif self.stream_type == 'RandomRBF':
            return RandomRBF(seed_model=seed, seed_sample=seed, **self.preinitialized_params)
        elif self.stream_type == 'Sine':
            return Sine(seed = seed, **self.preinitialized_params)
        elif self.stream_type == 'Hyperplane':
            return Hyperplane(seed = seed, **self.preinitialized_params)

        elif self.stream_type == 'Waveform':
            return Waveform(seed = seed, **self.preinitialized_params)
        elif self.stream_type == 'Sine':
            return Sine(seed = seed, **self.preinitialized_params)
        elif self.stream_type == 'SEA':
            return SEA(seed = seed, **self.preinitialized_params)
        elif self.stream_type == 'STAGGER':
            return STAGGER(seed = seed, **self.preinitialized_params)
        elif self.stream_type == 'Friedman':
            return Friedman(seed = seed)
        elif self.stream_type == 'Mv':
              return Mv(seed = seed)
        else:
            raise ValueError(f"Unknown stream type: {self.stream_type}")
        
# Create a class to abstract away the actions. It should enable each ActionType to be parameterized
# An ActionType comprises a method to be called and the parameters to be passed to the method
