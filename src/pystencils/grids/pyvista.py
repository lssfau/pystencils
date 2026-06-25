from __future__ import annotations

from .tensor_field import TensorField
from .patch import PatchGrid, VariablePlacement
from .patch_data import PatchData
from .protocols import ViewNdArray

try:
    import pyvista as pv

    HAS_PYVSTA = True
except ImportError:  # pragma: no cover
    HAS_PYVSTA = False


class PatchDataPyVistaBridge:
    def __init__(self, pdata: PatchData):
        if not HAS_PYVSTA:
            raise RuntimeError("Cannot create PyVista bridge: PyVista is not installed")

        self._pdata = pdata

    def get_mesh(self) -> pv.ImageData:
        mesh = self._create_empty_mesh()
        self._update_mesh_data(mesh)
        return mesh

    def plot_field(self, f: TensorField, *, component: int | None = None):
        mesh = self.get_mesh()

        pl = pv.Plotter()
        pl.add_mesh(mesh, scalars=f.name, component=component)
        pl.show()

    def _create_empty_mesh(self) -> pv.ImageData:
        mesh = pv.ImageData()
        mesh.dimensions = tuple(
            self._pdata.num_vertices,
        ) + (
            1,
        ) * (3 - self._pdata.dimensionality)
        mesh.origin = tuple(self._pdata.x_min) + (0.0,) * (
            3 - self._pdata.dimensionality
        )
        mesh.spacing = tuple(self._pdata.spacing) + (0.0,) * (
            3 - self._pdata.dimensionality
        )

        return mesh

    def _update_mesh_data(self, mesh: pv.ImageData):
        for k in self._pdata.data:
            match k:
                case TensorField():
                    v = self._pdata.asnumpy(k)
                    if isinstance(k, ViewNdArray):
                        v = k.view_ndarray(v)

                    for coord in range(self._pdata.dimensionality // 2):
                        v = v.swapaxes(coord, self._pdata.dimensionality - coord - 1)

                    match k.grid:
                        case PatchGrid(_, VariablePlacement.VERTICES):
                            mesh.point_data[k.name] = v.reshape(
                                (mesh.n_points,) + k.tensor_shape
                            )
                        case PatchGrid(_, VariablePlacement.CELLS):
                            mesh.cell_data[k.name] = v.reshape(
                                (mesh.n_cells,) + k.tensor_shape
                            )
