import numpy as np

from river import tree
from river.datasets.synth import RandomRBF, RandomTree, Sine, Hyperplane, Waveform, SEA, STAGGER, Friedman, Mv, Planes2D
from river.datasets import ImageSegments
from collections import OrderedDict
from actions import MultiplyAction

BINS = [1, 2, 3, 4]
NUM_STATES = len(BINS) * len(BINS)


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

    def update_delta(self, new_delta):
        self.delta = new_delta

    # Now remove the split update mechanism and compare with second best instead of current best
    # Let's begin by removing the update mechanism
    # The superclass tree.ExtremelyFastDecisionTreeClassifier has a method called    def _reevaluate_best_split(self, node, parent, branch_index, **kwargs):
    # It reevaluates the best split for the node.
    # We can override this method to remove the update mechanism
    # For the RL agent, the action is to choose whether to use the overriden update mechanism or the original one... strategy superposition (rather than alternatives)

    def _reevaluate_best_split(self, node, parent, branch_index, **kwargs):
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

    def _attempt_to_split(self, node, parent, branch_index, **kwargs):
        ''' 
            Override EFDT split Call HoeffdingTreeClassifier's _attempt_to_split method, in order to compare only with the second best split
            HoeffdingTreeClassifier is cutEFDT's superclass EFDT's superclass
        '''

        # Explicitly call the great-grandparent class method, because using Python's super in series didn't seem to work
        tree.HoeffdingTreeClassifier._attempt_to_split(self, node, parent, branch_index, **kwargs)


# Create a dictionary mapping class names to class objects
model_classes = {
    'UpdatableHoeffdingTreeClassifier': UpdatableHoeffdingTreeClassifier,
    'UpdatableEFDTClassifier': UpdatableEFDTClassifier,
    'CutEFDTClassifier': CutEFDTClassifier
}

# Create a dictionary mapping class names to their respective action space
# The action space is a method name within the class
action_spaces = {
    'UpdatableHoeffdingTreeClassifier': set([MultiplyAction(1, 1e-10, 1), MultiplyAction(100, 1e-10, 1), MultiplyAction(1/10, 1e-10, 1)]),
    'UpdatableEFDTClassifier': set([MultiplyAction(1, 1e-10, 1), MultiplyAction(100, 1e-10, 1), MultiplyAction(1/10, 1e-10, 1)]),
    'CutEFDTClassifier': ['MultiplyAction', '_reevaluate_best_split', '_attempt_to_split'] # TODO Fix this
}

class Environment:
    def __init__(self, model, model_baseline, stream_factory, delta_multipliers, num_samples_per_epoch, num_epochs_per_episode):
        self.model = model
        self.model_baseline = model_baseline

        self.current_episode = 0
        self.stream_factory = stream_factory
        self.stream = stream_factory.create(self.current_episode)

        self.state = None  # Initialize state - 0 indicates no change in accuracy

        self.delta_multipliers = delta_multipliers

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
        self.state = None # Initialize state - 0 indicates no change in accuracy
        self.stream = self.stream_factory.create(seed=self.current_episode) # A new seed for the stream

        self.cumulative_accuracy = 0.0
        self.cumulative_baseline_accuracy = 0.0

        # Return the initial state
        return self.state

    def step(self, delta_multiplier_idx):
        ############################ TODO: Update this to allow for multiple action types ############################


        # Update delta_hard based on the chosen delta_multiplier
        self.model.delta = self.action_update_delta_hard(delta_multiplier_idx)

        # Run one epoch of the experiment
        accuracy, baseline_epoch_prequential_accuracy = self.run_one_epoch()

        # Update cumulative prequential accuracy and cumulative baseline prequential accuracy
        self.cumulative_accuracy += accuracy
        self.cumulative_baseline_accuracy += baseline_epoch_prequential_accuracy

        # Calculate the reward as the difference between the prequential accuracy of the model obtained from reinforcement learning 
        # and that of the baseline model
        reward = accuracy - baseline_epoch_prequential_accuracy

        # Increment the epoch counter
        self.current_epoch += 1

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

        # Update the state with the accuracy changes
        self.state = self.index_state(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin)

        # Signal if the episode is done
        done = self.current_epoch == self.num_epochs

        return self.state, reward, done

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
    def index_state(epoch_accuracy_change_bin, epoch_5_accuracy_change_bin):
        # We take epoch_accuracy_change_bin and epoch_5_accuracy_change_bin and return a state index in the range [0, 24]
        # The state index is calculated as below:
        state_index = (epoch_accuracy_change_bin - 1) * len(BINS) + (epoch_5_accuracy_change_bin - 1) # 0, 5, 10, 15, 20 correspond to increasingly large 
        return state_index

    def action_update_delta_hard(self, multiplier_idx):
        # Adjust delta_hard based on the chosen multiplier index
        multiplier = self.delta_multipliers[multiplier_idx]

        delta = self.model.delta * (multiplier)

        # Ensure delta_hard is within the range [1e-10, 1]
        if delta < 1e-10:
            return 1e-10
        elif delta > 1:
            return 1

        return delta

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
        total_baseline_correctly_classified = 0
        total_samples = 0

        # Iterate over the data in the stream
        for x, y in self.stream.take(self.num_samples_per_epoch):

            # Predict the output for the current input
            prediction = self.model.predict_one(x)
            baseline_prediction = self.model_baseline.predict_one(x)

            # Learn from the current input-output pair
            self.model.learn_one(x, y)
            self.model_baseline.learn_one(x, y)

            # Calculate the accuracy of the prediction against the actual output
            is_correctly_classified = self.correctly_classified(prediction, y)
            is_baseline_correctly_classified = self.correctly_classified(baseline_prediction, y)

            # Add the correctly classified to the total correctly classified
            total_correctly_classified += is_correctly_classified
            total_baseline_correctly_classified += is_baseline_correctly_classified

            # Increment the total number of samples
            total_samples += 1

        # Calculate the prequential accuracy for this epoch
        epoch_prequential_accuracy = self.average_classification_accuracy(total_correctly_classified, total_samples)
        baseline_epoch_prequential_accuracy = self.average_classification_accuracy(total_baseline_correctly_classified, total_samples)

        # Return the prequential accuracy and baseline_epoch_prequential_accuracy
        return epoch_prequential_accuracy, baseline_epoch_prequential_accuracy

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
