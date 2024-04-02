"""
Individual runners i.e. classes that execute epochs in a training process.

"""
import array
import math
import statistics
import typing
import abc
import torch

from torch.utils import data
from torch import optim
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from sr_core.training import observers, events

Criterion = nn.Module


class EpochRunner(abc.ABC):
    """
    Base class defining an interface for running a single pass through the whole input dataset
    """

    @abc.abstractmethod
    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int) -> float:
        """
        :param model: Model to run epoch on.
        :param data_loader: Loader return batches of tuples (input, label)
        :return: Epoch loss
        """


class TrainingEpochRunner(EpochRunner, observers.Observable):
    """
    Runs training procedure, executing backward propagation.
    Observers are notified.
    """

    def __init__(self,
                 optimizer: optim.Optimizer,
                 criterion: Criterion,
                 device: str = 'cpu',
                 post_prediction_func: typing.Callable = None,
                 mixed_precision_training: bool = False,
                 **kwargs):
        observers.Observable.__init__(self)
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
        self._kwargs = kwargs
        self._post_prediction_func = post_prediction_func
        self.mixed_precision = mixed_precision_training
        self.scaler = GradScaler(enabled=mixed_precision_training)

    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int):
        model.train(True)
        batch_losses = array.array('d', [])
        self.notify(events.EpochEvents.EPOCH_STARTED, (epoch, len(data_loader.dataset), data_loader.batch_size))
        for batch_iterator, entry in enumerate(data_loader):
            self._optimizer.zero_grad()
            with autocast(self.mixed_precision):
                entry.to(self._device)
                output = model(entry)
                if self._post_prediction_func is not None:
                    output = self._post_prediction_func(lr=entry.lrs, sr=output, hr=entry.hr,
                                                        device=self._device, **self._kwargs)
                loss = self._criterion(output, entry.hr, entry.hr_mask)
            self.scaler.scale(loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
            if not math.isnan(loss):
                batch_losses.append(loss)
            self.notify(events.EpochEvents.BATCH_FINISHED, [epoch, batch_iterator,
                                                            statistics.mean(batch_losses) if batch_losses else 0.0])
            if batch_iterator > 10:
                break
        results = statistics.mean(batch_losses) if batch_losses else 0.0
        self.notify(events.EpochEvents.EPOCH_FINISHED, None)
        return results


class ValidationEpochRunner(EpochRunner, observers.Observable):
    """
    Passes through validation dataset without backward propagation.
    Observers are notified.
    """

    def __init__(self,
                 criterion: Criterion,
                 device: str = 'cpu'):
        observers.Observable.__init__(self)
        self._criterion = criterion
        self._device = device

    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int):
        model.train(False)
        batch_losses = array.array('d', [])
        self.notify(events.EpochEvents.EPOCH_STARTED, (epoch, len(data_loader.dataset), data_loader.batch_size))
        with torch.no_grad():
            for batch_iterator, entry in enumerate(data_loader):
                entry = entry.to(self._device)
                output = model(entry)
                self.notify(events.EpochEvents.VALIDATION_BATCH_OUTPUT,
                            observers.EpochValidationResults(epoch, batch_iterator, output))
                loss = self._criterion(output, entry.hr, hr_mask=entry.hr_mask)
                if not math.isnan(loss):
                    batch_losses.append(loss)
                self.notify(events.EpochEvents.BATCH_FINISHED, [epoch, batch_iterator,
                                                                statistics.mean(batch_losses) if batch_losses else 0.0])
            results = statistics.mean(batch_losses) if batch_losses else 0.0
            self.notify(events.EpochEvents.EPOCH_FINISHED, None)
        return results
