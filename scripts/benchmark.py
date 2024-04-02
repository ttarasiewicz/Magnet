import numpy as np
import cv2
import os
from pathlib import Path
import shutil
import sr_core.loss
from sr_core.loss import metrics
from sr_core.data.dataset import PatchConfig, PatchedDataset, SISRDataset, GraphDataset, StaticImageDataset
from sr_core.data.dataset import configurators
from sr_core.data import graph_builder as gb
from sr_core.constants import Subsets
from sr_core.data import collation
from sr_core.data.dataloader import get_loader
from sr_core import models
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import array
import statistics
import torchdata as td
import torch_geometric as tg
import matplotlib.pyplot as plt
from image_similarity_measures.quality_metrics import rmse, psnr, ssim, uiq


def standardize(image: np.ndarray):
    if image.min() == image.max():
        return np.zeros(image.shape)
    return (image - image.min()) / (image.max() - image.min()) * 255


def save_image(image: np.ndarray, save_dir, file_it):
    file_path = os.path.join(save_dir, f'{str(file_it).zfill(5)}.png')
    image = standardize(image)
    cv2.imwrite(file_path, image)

model_dir = Path('../logs/RAMS_p32_b32_cMSE_ProbaV_20210531-115128')
model_name = model_dir.parts[-1]
dataset_root = r'D:\deepsent\deepsent\dataset'
dataset_name = 'ProbaV'
subdataset_name = ''
input_images_count = 9
device = 'cuda:0'
model = models.RAMS()
model.eval()

print("Learnable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# quit()
radius = 1.0
is_sisr = False
recache = True
# model_name=f'{model_name}_r{str(radius).replace(".", ",")}'

print(model_name)
predictions_dir = '../predictions'
configurator = configurators.ProbaV
configurator = configurator(root=dataset_root, dataset_name=dataset_name, bands='NIR')

metrics = [metrics.cBase(loss) for loss in [rmse, psnr, ssim]]

patch_size = None
scale = 3

try:
    model.load_state_dict(torch.load(os.path.join(model_dir, 'weights')))
except Exception as e:
    raise e
model.eval()
model.to(device)
# model = models.BicubicTorch(device=device)

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

  # , get_lr_translations=False)
out_dir = os.path.join(predictions_dir, configurator.name, subdataset_name, f"{input_images_count}_lrs")
os.makedirs(out_dir, exist_ok=True)
print("Loading dataset")
test_dataset = configurator.get_dataset(Subsets.VALID, dataset_name=subdataset_name, register_lrs=True,
                                        register_mode='int')
print(f"Examples: {len(test_dataset)}")
if patch_size is not None:
    lr_config = PatchConfig((patch_size, patch_size))
    hr_config = PatchConfig((patch_size * scale, patch_size * scale))
    patched_dataset = PatchedDataset(hr_config, lr_config, same_shapes=True)
    test_dataset = patched_dataset(test_dataset)
    print(f"Patches: {len(test_dataset)}")

collate_fn = collation.MultipleImageCollation(num_lrs=input_images_count, permute=False)
if is_sisr:
    test_dataset = SISRDataset(test_dataset)
    # graph_builder = gb.FullGridBuilder()
    print("Train SISR dataset created")
    collate_fn = collation.SingleImageCollation()
if isinstance(model, models.GraphModel):
    graph_builder = gb.RadiusBuilder(radius=radius, lr_images=input_images_count)
    test_dataset = GraphDataset(test_dataset, graph_builder)

cache_path = Path('cache_benchmark')
if recache and cache_path.is_dir():
    shutil.rmtree(cache_path, ignore_errors=True)
# test_dataset.cache(cacher=td.cachers.Pickle(cache_path))

# loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
loader = get_loader(model, test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

test_loss = array.array('d', [])
loss_fn = sr_core.loss.cMSE()

hr_path = Path(out_dir)/'HR'
if not hr_path.is_dir():
    hr_path.mkdir(exist_ok=True, parents=True)
    from sr_core.data.entry import GraphEntry
    data: GraphEntry
    for i, data in enumerate(tqdm(loader, leave=True, desc='HR')):
        hr = data.hr.detach().cpu().numpy().squeeze()
        save_image(hr, hr_path, data.name[0])

model_prediction_dir = os.path.join(out_dir, model_name)
os.makedirs(model_prediction_dir, exist_ok=True)
metrics = {metric: array.array('d', []) for metric in metrics}
for i, data in enumerate(tqdm(loader, leave=True, desc=model_name)):
    data.to(device)
    with torch.no_grad():
        predicted = model(data)
    for func in metrics.keys():
        loss = func(predicted, data.hr)
        metrics[func].append(loss)
        # if func.name == ssim.__name__:
        #     print(i, round(loss, 4), round(statistics.mean(metrics[func]), 4))
    predicted = predicted.squeeze().detach().cpu().numpy()
    save_image(predicted, model_prediction_dir, data.name[0])

for func, loss in metrics.items():
    print(f"{func.name}: {statistics.mean(loss)}")

# Save results for each image in CSV file
import csv, json
with open(os.path.join(model_prediction_dir, '_scores.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([key.name for key in metrics.keys()])
    writer.writerows(zip(*metrics.values()))


# Save mean results of a model to JSON file
json_filename = os.path.join(out_dir, '_scores.json')
file_dict = {}
if os.path.isfile(json_filename):
    with open(json_filename) as f:
        file_dict = json.load(f)
if model_name not in file_dict.keys():
    file_dict[model_name] = {}
for func, loss in metrics.items():
    file_dict[model_name][func.name] = round(statistics.mean(loss), 6)
with open(json_filename, 'w') as file:
    json.dump(file_dict, file, indent=4, sort_keys=True)
print(json.dumps(file_dict, indent=4, sort_keys=True))
