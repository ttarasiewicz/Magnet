"""
Module with trainers i.e. classes responsible for training of a model.
Classes defined here are responsible for passing through the dataset
using training and validation runners defined in superdeep.runners package.
Observers are called with various events.
"""
import typing
import abc
from torch.utils import data
from torch import optim
from torch import nn
from sr_core.training import runners, stop, events, observers

LogPoint = observers.LogPoint
DataToSave = observers.AfterEpochDataToSave


class _NoScheduler:
    def step(self):
        pass


Criterion = nn.Module
Scheduler = typing.Union[_NoScheduler,
                         optim.lr_scheduler.StepLR,
                         optim.lr_scheduler.MultiStepLR,
                         optim.lr_scheduler.CosineAnnealingLR,
                         optim.lr_scheduler.ExponentialLR,
                         optim.lr_scheduler.LambdaLR]


class Trainer(abc.ABC, observers.Observable):
    @abc.abstractmethod
    def perform_training(self, epochs: int, training_loader, validation_loader):
        """
        Performs training of a model.
        """


class PyTorchTrainer(Trainer):
    """Generalized approach for training pytorch model."""

    def __init__(self, model: nn.Module,
                 training_epoch_runner: runners.EpochRunner,
                 validation_epoch_runner: runners.EpochRunner,
                 scheduler: Scheduler = _NoScheduler(),
                 stop_condition: stop.StopCondition = stop.NoCondition(),
                 initial_epoch: int = 0
                 ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.training_runner = training_epoch_runner
        self.validation_runner = validation_epoch_runner
        self.stop_condition = stop_condition
        self.initial_epoch = initial_epoch

    def perform_training(self, epochs: int,
                         training_loader: data.DataLoader,
                         validation_loader: data.DataLoader):
        self.notify(events.TraningEvents.TRAINING_BEGAN, None)
        for epoch in range(self.initial_epoch, epochs):
            self.scheduler.step()
            loss = self.training_runner.run_epoch(self.model, training_loader, epoch)
            self.notify(events.TraningEvents.TRAINING_EPOCH_FINISHED, [LogPoint("training_loss", loss, epoch)])

            val_loss = self.validation_runner.run_epoch(self.model, validation_loader, epoch)
            self.notify(events.TraningEvents.VALIDATION_EPOCH_FINISHED, [LogPoint("validation_loss", val_loss, epoch)])

            self.notify(events.TraningEvents.EPOCH_FINISHED, DataToSave(self.model, self.model.state_dict(),
                                                                        loss, val_loss, epoch))
            if self.stop_condition.is_condition_satisfied(epoch, loss, val_loss):
                break
