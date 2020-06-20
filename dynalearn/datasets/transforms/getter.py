from .transform import TransformList
from .random_flip import RandomFlipStateTransform
from .remap import RemapStateTransform

__transforms__ = {
    "RandomFlipStateTransform": RandomFlipStateTransform,
    "RemapStateTransform": RemapStateTransform,
}


def get(config):
    names = config.names
    transforms = []
    for n in names:
        if n in __transforms__:
            transforms.append(__transforms__[n](config))
        else:
            raise ValueError(
                f"{n} is invalid, possible entries are {list(__transforms__.keys())}"
            )
    return TransformList(transforms)
