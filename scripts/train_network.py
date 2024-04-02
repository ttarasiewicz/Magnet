import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import shutil
import pathlib
import time
from tqdm import tqdm
import statistics
import array
import tensorboardX
import cv2 as cv
from skimage import transform
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch_geometric as tg
from torch_geometric.data import Batch as tgBatch, DataLoader as tgDL
import torchdata as td

from sr_core.loss import MSE
from sr_core.data import collation, GraphEntry
from data.dataset import SISRDataset, PatchedDataset, PatchConfig, GraphDataset
from data.dataset import configurators
from sr_core.constants import Subsets
from sr_core.models.model import GraphModel
from sr_core import models
from sr_core import utils
from sr_core.data.graph_builder import RadiusBuilder


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def save_batch(predicted, data, epoch, batch, log_dir):
    def normalize(img):
        img = (img-img.min())/(img.max()-img.min())*255
        return img
    if isinstance(data, tgBatch):
        x = utils.graph_to_image2d(data.lrs, data.num_graphs, lr_shape=data.lr_shape[0])
        x = x[:1, ...]
    else:
        x = data.lrs[0, 0]
    x = x.detach().cpu().numpy().squeeze()
    x = normalize(x)
    x = transform.resize(x, (512, 512), order=0)
    predicted = predicted[0].detach().cpu().numpy().squeeze()
    predicted = normalize(predicted)
    predicted = transform.resize(predicted, (512, 512), order=0)
    target = data.hr[0].detach().cpu().numpy().squeeze()
    target = normalize(target)
    target = transform.resize(target, (512, 512), order=0)

    image = np.zeros((target.shape[0], target.shape[1] * 3))
    image[:, :target.shape[1]] = x
    image[:, target.shape[1]:2 * target.shape[1]] = predicted
    image[:, 2 * target.shape[1]:] = target
    cv.imwrite(os.path.join(log_dir, 'predictions', f'{str(epoch).zfill(3)}_{str(batch).zfill(4)}.png'), image)

# Batch(batch=[36864], edge_attr=[844544, 2], edge_index=[2, 844544], hr=[4, 1, 96, 96], hr_mask=[4, 1, 96, 96], lr_images=[4], lr_shape=[4, 2], lr_translations=[36, 2], lrs=[36864, 1], name=[4], pos=[36864, 2])
# torch.Size([1, 1, 32, 32])
model = models.MagNet_radius_reduce_res_preprocessed()
configurator = configurators.ProbaV(bands='NIR', root=r"E:\Programming\deepsent\dataset", name='ProbaV')
radius = 1.0
batch_size = 32

patch_size = 32
scale = 3
hr_size = patch_size * scale
lr = 1e-4
loss_fn = MSE()
register_images = True
recache = False
device_id = 0
load = True
last_epoch = 147
loaded_model_name = 'MagNet_radius_reduce_res_preprocessed_p32_b32_cMSE_ProbaV_20210606-210650'
log_dir = '../logs'

model_name = f'{model.name}_p{patch_size}_b{batch_size}_{loss_fn.name}_' \
             f'{configurator.name}_{time.strftime("%Y%m%d-%H%M%S")}'
if load:
    model_name = loaded_model_name

device = f'cuda:{device_id}'
print(model_name, device)
log_dir = os.path.join(log_dir, model_name)
if load:
    model.load_state_dict(torch.load(os.path.join(log_dir, 'weights')))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if load:
    optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'optimizer'), map_location=device))
    optimizer.param_groups[0]['lr'] = lr
logger = tensorboardX.SummaryWriter(os.path.join(log_dir, 'tensorlog'))
lr_config = PatchConfig((patch_size, patch_size))
hr_config = PatchConfig((patch_size * scale, patch_size * scale))
print("Loading dataset")
train_dataset = configurator.get_dataset(Subsets.TRAIN, register_lrs=register_images, register_mode='int')
print("Train examples: ", len(train_dataset))
train_dataset = PatchedDataset(hr_config, lr_config)(train_dataset)
print("Patched train examples: ", len(train_dataset))
val_dataset = configurator.get_dataset(Subsets.VALID, register_lrs=register_images, register_mode='int')
val_dataset = PatchedDataset(hr_config, lr_config)(val_dataset)

cache_path = os.path.join(f'cache{device_id}')
if os.path.isdir(cache_path) and recache:
    shutil.rmtree(cache_path)

if isinstance(model, GraphModel):
    train_dataset = GraphDataset(train_dataset, RadiusBuilder(radius=radius))
    val_dataset = GraphDataset(val_dataset, RadiusBuilder(radius=radius))
    train_dataset = train_dataset.cache(td.cachers.Pickle(pathlib.Path(os.path.join(cache_path, 'train'))))
    val_dataset = val_dataset.cache(td.cachers.Pickle(pathlib.Path(os.path.join(cache_path, 'val'))))
    loader = tgDL
    collation = None
    train_loader = loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = loader(val_dataset, batch_size=4)

else:
    loader = DataLoader
    collation = collation.MultipleImageCollation()
    train_dataset = train_dataset.cache(td.cachers.Pickle(pathlib.Path(os.path.join(cache_path, 'train'))))
    val_dataset = val_dataset.cache(td.cachers.Pickle(pathlib.Path(os.path.join(cache_path, 'val'))))
    train_loader = loader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collation)
    val_loader = loader(val_dataset, batch_size=4, collate_fn=collation)

print("Loaders initialized")
first_batch = next(iter(val_loader)).to(device)
print(first_batch)
fig = plt.figure(figsize=(16, 8))
grid = gridspec.GridSpec(2, 2)
axs = []
for gs in grid:
    axs.append(plt.subplot(gs))
grid.update(wspace=0.025, hspace=0.05)
plt.ion()
plt.show()

best_loss = None
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'predictions'), exist_ok=True)

print("Training...")
for epoch in range(last_epoch, 1000):
    train_losses = array.array('d', [])
    model.train()
    tqdm_generator = tqdm(train_loader, leave=True)
    result = 'NaN'
    for it, data in enumerate(tqdm_generator):
        tqdm_generator.set_description(f"Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']},"
                                       f" Loss: {result}")
        if 'hr_mask' not in data.__dict__.keys():
            data.hr_mask = None
        # for data in tqdm(train_loader, desc=f'T\\{epoch}'):
        # print(data.lrs.shape, data.hr.shape)
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(data.lrs[0].squeeze().numpy(), cmap='gray')
        # axs[1].imshow(data.hr[0].squeeze().numpy(), cmap='gray')
        # plt.show()
        data.to(device)
        optimizer.zero_grad()
        try:
            predicted = model(data)
        except Exception as e:
            print('!' * 40)
            print(data)
            raise e
        loss = loss_fn(predicted, data.hr, data.hr_mask)
        if torch.isnan(loss):
            print("NaN in loss function!")
            print(data.hr_mask)
            print(data.lr_translations)
            print(data)
            print(data.name)
            print(torch.isnan(predicted).any())
            quit()
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        result = statistics.mean(train_losses)

        if it % 20 == 0:
            test_batch = model(first_batch)
            save_batch(test_batch, first_batch, epoch, it, log_dir)
            for i, plot in zip(range(len(axs) // 2), axs):
                if i >= test_batch.shape[0]:
                    break
                plot.cla()
                plot.imshow(test_batch[i].detach().cpu().numpy().squeeze()[6:-6, 6:-6], cmap='gray',
                            interpolation='nearest')
            for i in range(len(axs) // 2, len(axs)):
                if i - len(axs) // 2 >= first_batch.hr.shape[0]:
                    break
                axs[i].cla()
                axs[i].imshow(first_batch.hr[i - len(axs) // 2].detach().cpu().numpy().squeeze()[6:-6, 6:-6],
                              cmap='gray')

            plt.suptitle(f"Epoch: {epoch}, Batch: {it}")
            plt.draw()
            plt.pause(0.1)

    # optimizer.param_groups[0]['lr'] = max([optimizer.param_groups[0]['lr'] * 0.9, 1e-4])
    train_losses = statistics.mean(train_losses) if train_losses else 0.0
    logger.add_scalar('train_loss', train_losses, epoch)
    val_losses = array.array('d', [0])
    model = model.eval()
    tqdm_generator = tqdm(val_loader, leave=True)
    result = 'NaN'
    for data in tqdm_generator:
        tqdm_generator.set_description(f"Epoch: {epoch}, Loss: {result}")
        if 'hr_mask' not in data.__dict__.keys():
            data.hr_mask = None
        data.to(device)
        with torch.no_grad():
            predicted = model(data)
            loss = loss_fn(predicted, data.hr, data.hr_mask)
        if torch.isnan(loss):
            for i in range(data.hr.shape[0]):
                print('-'*40)
                print(data.name[i])
                print(data.lrs.min(), data.lrs.max(), data.hr[i].min(), data.hr[i].max())
                print(predicted[i].min(), predicted[i].max())
                print(data.lr_translations)
                print(data.hr.shape)
                if data.hr_mask is not None:
                    print(data.hr_mask.sum())
                plt.figure()
                plt.imshow(data.hr[i].detach().cpu().numpy().squeeze(), cmap='gray')
                plt.figure()
                plt.imshow(predicted[i].detach().cpu().numpy().squeeze(), cmap='gray')
                plt.figure()
                plt.imshow(data.hr_mask[i].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
                plt.ioff()
                plt.show()
                plt.ion()
        val_losses.append(loss)
        result = statistics.mean(val_losses)

    val_losses = statistics.mean(val_losses)
    if best_loss is None:
        best_loss = val_losses
    if val_losses <= best_loss:
        best_loss = val_losses
        torch.save(model.state_dict(), os.path.join(log_dir, 'weights'))
        torch.save(model, os.path.join(log_dir, 'model'))
        torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer'))

    logger.add_scalar('val_loss', val_losses, epoch)
    print(f'Epoch: {epoch};\tT: {train_losses}; V: {val_losses}')
    time.sleep(0.1)
