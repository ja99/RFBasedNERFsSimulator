from typing import Optional
import pyvista as pv
import torch
import plotly.express as px
import numpy as np
import Sensor


def subsample(data: torch.Tensor, subsample_rate: int) -> torch.Tensor:
    data = torch.unsqueeze(data, 0)
    data = torch.nn.AvgPool3d(subsample_rate, subsample_rate)(data)[0]
    return data


#ToDo: make real time possible
#ToDo: add a slider to step through the time steps
def visualize_e_field(data: np.ndarray, subsampling: Optional[int]):
    # opacity = [0, 0.01, 0.2, 0.7, 1.0]

    if subsampling:
        data = subsample(torch.tensor(data), subsampling).cpu().numpy()

    data = pv.wrap(data)
    # data.plot(volume=True, opacity=opacity)  # Volume render
    # plotter=pv.Plotter()
    # plotter.add_volume(data, opacity='sigmoid_10')
    # plotter.show(auto_close=False)
    data.plot(volume=True, show_bounds=True)  # Volume render

def visualize_objs(
        export_grid: np.ndarray,  # [x, y, z, permittivity, conductivity]
        subsampling: Optional[int]
):
    # opacity = [0, 0.01, 0.2, 0.7, 1.0]

    export_grid = export_grid[:, :, :, 0]

    if subsampling:
        export_grid = subsample(torch.tensor(export_grid), subsampling).cpu().numpy()

    export_grid = pv.wrap(export_grid)
    # export_grid.plot(volume=True, opacity=opacity)  # Volume render
    export_grid.plot(volume=True, show_bounds=True)  # Volume render


def visualize_sensor(sensor: Sensor.SensorDevice):
    locations = np.stack([antenna.position for antenna in sensor.antennas])
    px.scatter_3d(
        x=locations[:, 0],
        y=locations[:, 1],
        z=locations[:, 2],
    ).show()