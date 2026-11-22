from dataclasses import dataclass

from SDF import SDFObject


@dataclass
class WorldObject:
    sdf: SDFObject
    permittivity: float
    conductivity: float
