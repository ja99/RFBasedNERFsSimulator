from typing import Union, List
from dataclasses import dataclass
import torch
import SDF


@dataclass
class WorldObject:
    sdf: Union[SDF.SDFObject]
    permittivity: torch.tensor
    conductivity: torch.tensor



