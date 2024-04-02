"""
Stop conditions used while training procedure to end the training
if loss is not improving. Could be used apart from epochs count only.

"""
import abc
import array


class StopCondition(abc.ABC):
    @abc.abstractmethod
    def is_condition_satisfied(self, epoch: int, train_epoch_loss: float, validation_epoch_loss: float) -> bool:
        pass


class NoCondition(StopCondition):
    """Condition that is never satisfied."""

    def is_condition_satisfied(self, epoch: int, train_epoch_loss: float, validation_epoch_loss: float) -> bool:
        return False


class ValidationLossDidNotImprove(StopCondition):
    """
    Condition that checks whether the validation loss got reduced by a defined value during n last epochs.
    """

    def __init__(self, patience: int, min_delta: float):
        """

        :param patience: Amount of last epochs under consideration.
        :param min_delta: What is the minimum value by which loss needs to change.
        """
        self._patience = patience
        self._min_delta = min_delta
        self._validation_losses = array.array('d', [])

    def is_condition_satisfied(self, epoch: int, train_epoch_loss: float, validation_epoch_loss: float) -> bool:
        self._validation_losses.append(validation_epoch_loss)
        if len(self._validation_losses) >= self._patience:
            oldest = self._validation_losses[-self._patience]
            differences = [oldest - value for value in self._validation_losses[-self._patience + 1:]]
            improved = map(lambda x: x > self._min_delta, differences)
            return not any(improved)
        return False
