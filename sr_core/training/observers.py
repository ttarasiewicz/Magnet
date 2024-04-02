"""
Observers - object that wait for particular event and somehow response to them.
Most of observers that can be found in this module are use to log information about traning process.

"""
import abc
import os
import csv
import time
import collections
import statistics
import torch
import math
import numpy as np
import cv2
from typing import Any, List, Union
import matplotlib.pyplot as plt
from torch.utils import data
from sr_core.training import events
import neptune.new as neptune
from neptune.new.types import File
import wandb

Path = Union[os.PathLike, str]


class Observer(abc.ABC):
    @abc.abstractmethod
    def on_event(self, event, data: Any):
        """
        Callback which is called every time observed object fires an event.

        :param event: Event to handle
        :param data: Data associated with event
        """


class Observable:
    """
    Objects that are observable can notify different observers of specific
    events.

    """

    def __init__(self):
        self.observers = []

    def add_observer(self, observer: Observer):
        """
        Adds observer so it can be notified of events.

        """
        self.observers.append(observer)

    def notify(self, event, data: Any):
        for observer in self.observers:
            observer.on_event(event, data)

AfterEpochDataToSave = collections.namedtuple("DataToSave", ["model", "state_dict",
                                                             "train_loss", "val_loss", "epoch"])

LogPoint = collections.namedtuple("LogPoint", ["name", "value", "epoch"])

EpochValidationResults = collections.namedtuple("EpochValidationResults",
                                                ["epoch", "batch", "data"])

class PyTorchModelSaver(Observer):
    MODEL_FILENAME = "model"
    WEIGHTS_FILENAME = "weights"
    OPTIM_FILENAME = "optimizer"

    def __init__(self, log_directory: Path, mode='V', ):
        self.log_directory = os.path.join(log_directory, 'saved_states')
        os.makedirs(self.log_directory, exist_ok=True)
        self.mode = mode.lower()

    def on_event(self, event, data: AfterEpochDataToSave):
        if event == events.TraningEvents.EPOCH_FINISHED:
            self.save_states(data, 'last')
            if self.best_loss is None or self.best_loss > data.val_loss:
                self.best_loss = data.val_loss
                self.save_states(data, 'best')

    def _get_subdir_paths(self, subdir):
        return os.path.join(self.log_directory, f"{subdir}_{self.MODEL_FILENAME}"),\
               os.path.join(self.log_directory, f"{subdir}_{self.WEIGHTS_FILENAME}"),\
               os.path.join(self.log_directory, f"{subdir}_{self.OPTIM_FILENAME}")

    def save_states(self, data: AfterEpochDataToSave, subdir):
        model_path, weights_path, optim_path = self._get_subdir_paths(subdir)
        with open(model_path, 'wb') as file:
            torch.save(data.model, file)
        with open(weights_path, 'wb') as file:
            torch.save(data.state_dict, file)
        with open(optim_path, 'wb') as file:
            torch.save(data.optimizer.state_dict(), file)


class FirstValidationBatchSaver(Observer):
    class ResultsReshaper:
        def reshape_to_HWC(self, data: torch.Tensor) -> List[np.ndarray]:
            return []

    class NCHWReshaper(ResultsReshaper):
        def __init__(self, rescaling_factor: float = 1.0):
            self._rescaling_factor = rescaling_factor

        def reshape_to_HWC(self, tensor: torch.Tensor) -> List[np.ndarray]:
            tensor = tensor.cpu()
            result = []
            for img_idx in range(0, tensor.shape[0]):
                result.append(
                    np.reshape(tensor[img_idx].numpy(),
                               [tensor.shape[2],
                                tensor.shape[3],
                                tensor.shape[1]]) * self._rescaling_factor
                )
            return result

    SUBDIR_NAME = "validation"

    def __init__(self):
        pass

    def on_event(self, event, data: EpochValidationResults):
        if event == events.EpochEvents.VALIDATION_BATCH_OUTPUT:
            if data.batch == 0:
                wandb.log({"val_pred": wandb.Image(data.data.cpu()[0, 0])})


# class ConfigManager(Observer):
#
#     def __init__(self, config_file: Config):
#         self.config_file = config_file
#         self.best_loss = config_file.best_loss
#
#     def on_event(self, event, data: AfterEpochDataToSave):
#         if event == events.TraningEvents.EPOCH_FINISHED:
#             self.config_file.last_epoch = data.epoch
#             if self.best_loss is None or data.val_loss < self.best_loss:
#                 self.best_loss = data.val_loss
#                 self.config_file.best_epoch = data.epoch
#                 self.config_file.best_loss = data.val_loss
#             self.config_file.save()
#
#
# class KerasModelSaver(Observer):
#     DataToSave = collections.namedtuple("DataToSave",
#                                         ["file_name", "model"])
#
#     def __init__(self, log_directory: Path):
#         self.log_directory = log_directory
#
#     def on_event(self, event, data: DataToSave):
#         if event == events.TraningEvents.EPOCH_FINISHED:
#             for data_point in data:
#                 path = os.path.join(self.log_directory, data_point.file_name)
#                 data_point.model.save(path)


class LossLogger(Observer):

    def __init__(self):
        pass

    def on_event(self, event, data: AfterEpochDataToSave):
        if event == events.TraningEvents.EPOCH_FINISHED:
            wandb.log({'validation_loss': data.val_loss})
            wandb.log({'training_loss': data.train_loss})
            torch.save(data.state_dict, 'model_dict.ckpt')
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file('model_dict.ckpt')
            wandb.run.log_artifact(artifact)

class ProgressLogger(Observer):
    class ProgressBar:
        def __init__(self, total: int, prefix: str):
            self._total = total
            self._iter = 0
            self._prefix = prefix
            self._started = time.time()
            if total != 0:
                self.tick()

        def tick(self, loss=None):
            length = 100
            decimals = 1
            current = min(100.0, 100.0 * self._iter / float(self._total))
            percent = ("{0:." + str(decimals) + "f}").format(current)
            filledLength = int(current)
            bar = 'X' * filledLength + '_' * (length - filledLength)
            # print('\r%s %s |%s| %s%% %s' % (self.prefix, f'({self._iter}\\{self._total})', bar, percent, "Completed"),
            #       end='', flush=True)
            elapsed_time = time.time() - self._started
            if self._iter != 0:
                seconds_per_iteration = round(elapsed_time / self._iter, 2)
                time_left = seconds_per_iteration * (self._total - self._iter)
                time_left = time.strftime("%M:%S", time.gmtime(round(time_left, 2)))
            else:
                seconds_per_iteration = '-'
                time_left = '-'

            print(f'\r{self._prefix} ({self._iter}\\{self._total}) |{bar}| {percent}% '
                  f'[{time.strftime("%M:%S", time.gmtime(elapsed_time))}<{time_left}, {seconds_per_iteration}s/it] ' +
                  f'Loss: {loss}',
                  end='', flush=True)
            self._iter = self._iter + 1

        def finish(self):
            print()

    def __init__(self, name: str):
        self._name = name
        self._pb = None

    def on_event(self, event, data: Any):
        if event == events.EpochEvents.EPOCH_STARTED:
            self._pb = ProgressLogger.ProgressBar(math.ceil(float(data[1]) / float(data[2])),
                                                  f"{self._name}({str(data[0]).zfill(4)}):")
        elif event == events.EpochEvents.BATCH_FINISHED:
            self._pb.tick(data[-1])
        elif event in events.EpochEvents.EPOCH_FINISHED:
            self._pb.finish()


# class TrainingVisualizer(Observer):
#     def __init__(self, model, data_loader: data.DataLoader, log_dir, device='cpu', save=True, display_step=50,
#                  epoch=0, window_display=True):
#         os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#         self.window_display = window_display
#         self.epoch = epoch
#         self._model = model
#         self._device = device
#         data_loader = data.DataLoader(data_loader.dataset,
#                                       shuffle=False,
#                                       collate_fn=data_loader.collate_fn,
#                                       batch_size=1)
#         data_loader = iter(data_loader)
#         self.entry = next(data_loader)
#         if isinstance(self.entry, SREntry):
#             self.hr_image = self.entry.hr_image.detach().cpu().numpy().squeeze()
#             self.lr = self.entry.lr_images[0, 0, ...].detach().cpu().numpy().squeeze()
#             self._on_event = self._predict_srentry
#         elif isinstance(self.entry, MultiBandSREntry):
#             self.output_band = self.entry.output_bands[0]
#             self.hr_image = self.entry.hr_image[self.output_band][0, ...].numpy().squeeze()
#             self.lr = self.entry.lr_images[self.output_band][0, 0, ...].numpy().squeeze()
#             self._on_event = self._predict_multibandsrentry
#         else:
#             print(f"Entry has to be of type SREntry or MultiBandSREntry, got {type(self.entry)}.")
#             exit()
#         self.bicubic = cv2.resize(self.lr, self.hr_image.shape, cv2.INTER_CUBIC)
#         self.entry.to(device)
#         self.display_step = display_step
#         self.save = save
#         self.log_dir = os.path.join(log_dir, 'predictions')
#         if save:
#             os.makedirs(self.log_dir, exist_ok=True)
#         if self.window_display:
#             self.f, self.axs = plt.subplots(2, 2, figsize=(12, 7))
#             self.f.canvas.set_window_title(f'{log_dir}\t{self.entry.name}')
#             plt.ion()
#             plt.show()
#             self.show()
#
#     def _clear_axes(self):
#         for row in self.axs:
#             for plot in row:
#                 plot.cla()
#
#     def show(self, predicted=None, batch=0):
#         if not self.window_display:
#             return
#         if predicted is None:
#             predicted = np.zeros(self.hr_image.shape)
#         self._clear_axes()
#         self.f.suptitle(f"Epoch: {self.epoch}, Batch: {batch}")
#         self.axs[0][0].set_title("LR")
#         self.axs[0][1].set_title("HR")
#         self.axs[1][0].set_title("Bicubic")
#         self.axs[1][1].set_title("SR")
#         self.axs[0][0].imshow(self.lr, cmap="Greys_r", interpolation='none')
#         self.axs[0][1].imshow(self.hr_image, cmap="Greys_r", interpolation='none')
#         self.axs[1][0].imshow(self.bicubic, cmap="Greys_r",
#                               interpolation='none')
#         self.axs[1][1].imshow(predicted, cmap="Greys_r", interpolation='none', vmin=0., vmax=255.)
#         self.f.tight_layout(w_pad=0.1, pad=0.3)
#         self.f.subplots_adjust(top=0.95)
#         plt.draw()
#         plt.pause(0.001)
#
#     def on_event(self, event, data):
#         if event == events.EpochEvents.BATCH_FINISHED and data[1] % self.display_step == 0:
#             self._on_event(data)
#         if event == events.EpochEvents.EPOCH_FINISHED:
#             self.epoch += 1
#
#     def _predict_srentry(self, data):
#         with torch.no_grad():
#             prediction = self._model(self.entry)
#         prediction = self._normalize(prediction)
#         self.show(prediction, data[1])
#         if self.save:
#             hr_image = self.entry.hr_image.detach().cpu().numpy().squeeze()
#             hr_image = self._normalize(hr_image)
#             prediction = np.concatenate([prediction, hr_image], axis=1)
#             os.makedirs(os.path.join(self.log_dir), exist_ok=True)
#             filename = f"{str(data[0]).zfill(3)}_{str(data[1]).zfill(5)}.png"
#             cv2.imwrite(os.path.join(self.log_dir, filename), prediction)
#
#     def _predict_multibandsrentry(self, data):
#         with torch.no_grad():
#             predictions = self._model(self.entry)
#         for k in predictions.keys():
#             predictions[k] = self._normalize(predictions[k]).astype(int)
#         if self.save:
#             for k, v in predictions.items():
#                 hr_image = self.entry.hr_image[k].detach().cpu().numpy().squeeze()
#                 hr_image = self._normalize(hr_image)
#                 v = np.concatenate([v, hr_image], axis=1)
#                 os.makedirs(os.path.join(self.log_dir, k), exist_ok=True)
#                 filename = f"{str(data[0]).zfill(3)}_{str(data[1]).zfill(5)}.png"
#                 cv2.imwrite(os.path.join(self.log_dir, k, filename), v)
#         self.show(predictions[self.output_band], data[1])
#
#     def _normalize(self, image):
#         if isinstance(image, torch.Tensor):
#             image: np.ndarray = image.detach().cpu().numpy().squeeze()
#         image_cropped = image[3:-3, 3:-3, ...]
#         image = image - image_cropped.min()
#         image = image/(image_cropped.max() - image_cropped.min())
#         image = (image*255.).astype(int)
#         return image