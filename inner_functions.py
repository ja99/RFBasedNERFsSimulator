from typing import List, Tuple

import numpy as np
from torch import multiprocessing as mp
from tqdm import tqdm
import WorldObjectType


def min_dist_to_obj(objects: List[WorldObjectType.WorldObject], point: np.ndarray) -> (float, int):
    distances = [obj.sdf.sdf(point) for obj in objects]
    d = min(distances)
    obj_id = distances.index(d)
    return d, obj_id


def add_objects_to_grid_inner(objects: List[WorldObjectType.WorldObject],
                              objects_queue: mp.Queue,
                              subgrid_queue: mp.Queue,
                              kill_me_queue: mp.Queue,
                              subgrid_q_lock: mp.Lock,
                              kill_me_q_lock: mp.Lock,
                              process_id: int,
                              x_range: Tuple[int, int],
                              y_range: Tuple[int, int],
                              z_range: Tuple[int, int],
                              ):
    export_grid = np.zeros((x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0], 3))

    in_region = False
    current_obj = -1
    z_start = -1

    for x in tqdm(range(x_range[0], x_range[1]), desc=f"Process {process_id}"):
        for y in range(y_range[0], y_range[1]):
            for z in range(z_range[0], z_range[1]):
                d, obj_i = min_dist_to_obj(objects, np.array([x, y, z]))

                # outside the object
                if 0 < d:
                    export_grid[x - x_range[0], y - y_range[0], z - z_range[0]] = np.array([1.0, 1.0, d])
                    if in_region:
                        in_region = False
                        z_end = z
                        objects_queue.put(
                            (
                                x,
                                y,
                                z_start,
                                z_end,
                                f"{x},{y},{z_start}:{z_end},",
                                float(objects[current_obj].permittivity),
                                float(objects[current_obj].conductivity)
                            )
                        )
                # inside the object
                elif d < 0:
                    export_grid[x - x_range[0], y - y_range[0], z - z_range[0]] = np.array(
                        [float(objects[obj_i].permittivity),
                         float(objects[obj_i].conductivity),
                         d])

                    if not in_region:
                        in_region = True
                        current_obj = obj_i
                        z_start = z

                    if in_region and obj_i != current_obj:
                        z_end = z
                        objects_queue.put(
                            (
                                x,
                                y,
                                z_start,
                                z_end,
                                f"{x},{y},{z_start}:{z_end},",
                                float(objects[current_obj].permittivity),
                                float(objects[current_obj].conductivity)
                            )
                        )
                        current_obj = obj_i
                        z_start = z

                    if z == z_range[1] - 1:
                        z_end = z_range[1]
                        objects_queue.put(
                            (
                                x,
                                y,
                                z_start,
                                z_end,
                                f"{x},{y},{z_start}:{z_end},",
                                float(objects[current_obj].permittivity),
                                float(objects[current_obj].conductivity)
                            )
                        )
                        current_obj = -1
                        z_start = -1
                        in_region = False

    with subgrid_q_lock:
        subgrid_queue.put((x_range, y_range, export_grid))

    # somehow processes wouldn't die, so I had to do this
    with kill_me_q_lock:
        kill_me_queue.put(process_id)

    print(f"Process {process_id} finished")




