from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import numpy as np
from sr_core.data import Entry
import torch


def register_entry(entry: Entry, mode: str = 'int'):
    assert entry.lr_translations is not None, "No translations calculated for LR images!"
    assert mode not in ['none', 'None', False, None], f"Wrong registering mode: {mode}"
    import time
    time.sleep(0.2)
    for i in range(len(entry.lrs)):
        tr = entry.lr_translations[i]
        if mode == 'int':
            tr = np.round(tr)
        entry.lr_translations[i] = entry.lr_translations[i] - tr
        entry.lrs[i] = shift(entry.lrs[i], -tr, mode='reflect')
        if entry.lr_masks is not None:
            entry.lr_masks[i] = np.ceil(shift(entry.lr_masks[i], -tr, mode='constant', cval=0))
    return entry


def get_entry_translations(entry: Entry, mode: str, upsample_factor=100):
    assert mode in ['int', 'float'], f"Wrong mode ({mode}) - choose one of 'int' or 'float'!"
    ref = 0
    translations = []
    mode = 'float'
    if mode == 'int':
        if entry.lr_masks is not None:
            for i in range(len(entry.lrs)):
                shifts = -phase_cross_correlation(entry.lrs[ref], entry.lrs[i],
                                                  reference_mask=entry.lr_masks[ref], moving_mask=entry.lr_masks[i],
                                                  upsample_factor=upsample_factor, return_error=False)
                translations.append(shifts)
        else:
            for i in range(len(entry.lrs)):
                shifts = -phase_cross_correlation(entry.lrs[ref], entry.lrs[i], upsample_factor=1, return_error=False)
                translations.append(shifts)

    else:
        for i in range(len(entry.lrs)):
            shifts = -phase_cross_correlation(entry.lrs[ref], entry.lrs[i], upsample_factor=100, return_error=False)
            translations.append(shifts)
    print(translations)
    entry.lr_translations = translations
    return entry
