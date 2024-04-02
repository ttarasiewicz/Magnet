import enum


class Subsets(enum.Enum):
    TRAIN = 'train'
    VALID = 'val'
    TEST = 'test'


class EntryElements(enum.Enum):
    LRS = 'lrs'
    HR = 'hr'
    LR_MASKS = 'lr_masks'
    HR_MASK = 'hr_mask'
    TRANSLATIONS = 'lr_translations'


class Order(enum.Enum):
    ASC = True
    DESC = False
