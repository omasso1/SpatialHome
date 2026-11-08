from ursina import Vec3
import numpy as np

def norm(vec: Vec3) -> float:
    return float(np.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2))