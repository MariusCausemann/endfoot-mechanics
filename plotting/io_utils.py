import numpy as np
import yaml
import meshio
import vtk
import pyvista as pv


def read_xdmf_timeseries(filename, idx=None, variables=None):
    times = []
    point_dataset = []
    cell_data_set = []
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        if points.shape[1]==2:
            points = np.append(points, np.zeros((points.shape[0], 1)), axis=-1)
        if idx is None:
            idx = range(reader.num_steps)
        for k in idx:
            t, point_data, cell_data = reader.read_data(k)
            times.append(t)
            data_dict = dict()
            for name, data in point_data.items():
                if variables is not None:
                    if name in variables:
                        data_dict[name] = data
                else:
                        data_dict[name] = data
            point_dataset.append(data_dict)
            data_dict = dict()
            for name, data in cell_data.items():
                if variables is not None:
                    if name in variables:
                        data_dict[name] = data
                else:
                        data_dict[name] = data
            cell_data_set.append(data_dict)
            
    return times, points, cells, point_dataset, cell_data_set


def xdmf_to_unstructuredGrid(filename, idx=None, variables=None):
    times, points, cells, point_dataset, cell_dataset = read_xdmf_timeseries(filename, idx=idx,
                                                                             variables=variables)
    n = cells[0].data.shape[0] # number of cells
    p = cells[0].data.shape[1] # number of points per cell
    c = cells[0].data
    
    if (points[:,2]==0).all():
        cell_type = vtk.VTK_TRIANGLE
    else:
        cell_type = vtk.VTK_TETRA

    c = np.insert(c, 0, p, axis=1) # insert number of points per cell at begin of cell row
    grid = pv.UnstructuredGrid(c, np.repeat(cell_type, n), points)
    for i, p_data in enumerate(point_dataset):   
        if len(point_dataset)==1:
            array_name = "{name}"  
        else:
            array_name = "{name}_{idx}"  
        for name, data in p_data.items():
            if idx is None:
                current_idx = None
            else:
                current_idx = idx[i]
            grid.point_data[array_name.format(name=name, idx=current_idx)] = data

    for i, c_data in enumerate(cell_dataset):
        if len(cell_dataset)==1:
            array_name = "{name}"  
        else:
            array_name = "{name}_{idx}"  
        for name, data in c_data.items():
            if idx is None:
                current_idx = None
            else:
                current_idx = idx[i]
            #from IPython import embed; embed()
            grid.cell_data[array_name.format(name=name, idx=current_idx)] = data[0]

    return grid