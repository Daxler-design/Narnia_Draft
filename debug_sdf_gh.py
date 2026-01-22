from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import Rhino.Geometry as rg #type: ignore


PointTuple = Tuple[float, float, float]
PlaneTuple = Tuple[PointTuple, PointTuple, PointTuple]
PlaneLike = Union["rg.Plane", PlaneTuple]


# grasshopper CPython helper: load a slice from NPZ and return Rhino.Geometry curves
def npz_slice_to_rhino_curves(
    npz_path: str,
    field: str = "result_fields",
    slice_idx: int = 0,
    iso_level: float = 0.0,
) -> List["rg.PolylineCurve"]:

    """
    Input: npz_path (str), field (str), slice_idx (int), iso_level (float)
    Output: list of Rhino.Geometry.PolylineCurve.
    """
    import numpy as _np
    import Rhino.Geometry as _rg #type: ignore

    data = _np.load(npz_path, allow_pickle=True)
    fields = data[field]
    num_slices, values_per_field = fields.shape
    n = int(_np.sqrt(values_per_field))
    if n * n != values_per_field:
        raise ValueError(f"Cannot infer square grid from {values_per_field} values.")
    nx = ny = n

    if slice_idx < 0 or slice_idx >= num_slices:
        raise IndexError(f"slice_idx {slice_idx} out of range 0..{num_slices - 1}")

    if "bounds_min" in data:
        bmin = data["bounds_min"]
    else:
        bmin = _np.array([0.0, 0.0, 0.0])
    if "bounds_max" in data:
        bmax = data["bounds_max"]
    else:
        bmax = _np.array([1.0, 1.0, 1.0])
    xmin, ymin = float(bmin[0]), float(bmin[1])
    xmax, ymax = float(bmax[0]), float(bmax[1])

    slice_2d = fields[slice_idx].reshape((ny, nx))

    def _interp(p1, p2, v1, v2):
        if abs(v2 - v1) < 1e-12:
            t = 0.5
        else:
            t = (iso_level - v1) / (v2 - v1)
        return p1 + t * (p2 - p1)

    # Marching squares edges per cell
    segments = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            v00 = slice_2d[iy, ix]
            v10 = slice_2d[iy, ix + 1]
            v11 = slice_2d[iy + 1, ix + 1]
            v01 = slice_2d[iy + 1, ix]

            case = 0
            if v00 >= iso_level:
                case |= 1
            if v10 >= iso_level:
                case |= 2
            if v11 >= iso_level:
                case |= 4
            if v01 >= iso_level:
                case |= 8
            if case == 0 or case == 15:
                continue

            # cell corner coords in grid space
            p00 = _np.array([ix, iy], dtype=float)
            p10 = _np.array([ix + 1, iy], dtype=float)
            p11 = _np.array([ix + 1, iy + 1], dtype=float)
            p01 = _np.array([ix, iy + 1], dtype=float)

            # edge interpolation points
            e0 = _interp(p00, p10, v00, v10)  # top
            e1 = _interp(p10, p11, v10, v11)  # right
            e2 = _interp(p11, p01, v11, v01)  # bottom
            e3 = _interp(p01, p00, v01, v00)  # left

            # case table (segments)
            if case in (1, 14):
                segments.append((e3, e0))
            elif case in (2, 13):
                segments.append((e0, e1))
            elif case in (3, 12):
                segments.append((e3, e1))
            elif case in (4, 11):
                segments.append((e1, e2))
            elif case in (5, 10):
                segments.append((e3, e0))
                segments.append((e1, e2))
            elif case in (6, 9):
                segments.append((e0, e2))
            elif case in (7, 8):
                segments.append((e3, e2))

    if not segments:
        return []

    # Map grid coords to world XY
    def _to_world(pt):
        x = xmin + (pt[0] / (nx - 1)) * (xmax - xmin)
        y = ymin + (pt[1] / (ny - 1)) * (ymax - ymin)
        return _rg.Point3d(x, y, 0.0)

    # Stitch segments into polylines
    tol = 1e-6
    def _key(pt):
        return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

    seg_map = {}
    for a, b in segments:
        ka = _key(a)
        kb = _key(b)
        seg_map.setdefault(ka, []).append(b)
        seg_map.setdefault(kb, []).append(a)

    curves = []
    visited = set()
    for a, b in segments:
        ka = _key(a)
        kb = _key(b)
        if (ka, kb) in visited or (kb, ka) in visited:
            continue

        poly = [a, b]
        visited.add((ka, kb))

        # extend forward
        while True:
            last = poly[-1]
            kl = _key(last)
            neighbors = seg_map.get(kl, [])
            next_pt = None
            for npt in neighbors:
                kn = _key(npt)
                if (kl, kn) not in visited and (kn, kl) not in visited:
                    next_pt = npt
                    visited.add((kl, kn))
                    break
            if next_pt is None:
                break
            poly.append(next_pt)

        # extend backward
        while True:
            first = poly[0]
            kf = _key(first)
            neighbors = seg_map.get(kf, [])
            next_pt = None
            for npt in neighbors:
                kn = _key(npt)
                if (kn, kf) not in visited and (kf, kn) not in visited:
                    next_pt = npt
                    visited.add((kn, kf))
                    break
            if next_pt is None:
                break
            poly.insert(0, next_pt)

        if len(poly) >= 2:
            pts = [_to_world(p) for p in poly]
            curves.append(_rg.PolylineCurve(pts))

    return curves


Segment = Tuple[np.ndarray, np.ndarray]
Polyline = List[np.ndarray]


class SdfGhHelper:
    """
    Minimal GH helper for true SDF NPZ stacks (world units).
    Keeps reading, operations, curves, and mesh responsibilities separate.
    """

    def __init__(
        self,
        npz_path: str,
        field: str = "result_fields",
        iso_level: float = 0.0,
        smooth_method: str = "native",
        smooth_strength: float = 0.5,
        smooth_iters: int = 1,
        tol: float = 1e-6,
    ) -> None:
        self.npz_path = npz_path
        self.field = field
        self.iso_level = float(iso_level)
        self.smooth_method = smooth_method
        self.smooth_strength = float(smooth_strength)
        self.smooth_iters = int(smooth_iters)
        self.tol = float(tol)

        self.bounds_min = None
        self.bounds_max = None
        self.total_height = None
        self.slice_count = None
        self.slice_count_original = None
        self.slice_count_current = None
        self.nx = None
        self.ny = None
        self.grid_dx = None
        self.grid_dy = None
        self.slice_dz = None
        self.sdf_stack = None

    # I/O
    def load(self, max_slices: Optional[int] = None) -> "SdfGhHelper":
        """
        Input: max_slices (int or None)
        Output: SdfGhHelper (self), with sdf_stack and spacing populated.
        """
        data = np.load(self.npz_path, allow_pickle=True)
        fields = data[self.field]
        num_slices, values_per_field = fields.shape
        self.slice_count_original = int(num_slices)

        n = int(np.sqrt(values_per_field))
        if n * n != values_per_field:
            raise ValueError(
                "Cannot infer square grid from %s values." % values_per_field
            )
        self.nx = n
        self.ny = n

        if max_slices is not None:
            max_slices = int(max_slices)
            if max_slices <= 0:
                raise ValueError("max_slices must be > 0.")
            if max_slices < num_slices:
                num_slices = max_slices
                fields = fields[:num_slices]

        self.slice_count = int(num_slices)
        self.slice_count_current = int(num_slices)
        self.sdf_stack = fields.reshape(self.slice_count, self.ny, self.nx)

        if "bounds_min" in data:
            bmin = data["bounds_min"]
        else:
            bmin = np.array([0.0, 0.0, 0.0])
        if "bounds_max" in data:
            bmax = data["bounds_max"]
        else:
            bmax = np.array([1.0, 1.0, 1.0])
        self.bounds_min = np.array(bmin, dtype=float)
        self.bounds_max = np.array(bmax, dtype=float)
        if "total_height" in data:
            self.total_height = data["total_height"]
        else:
            self.total_height = None

        if self.bounds_min is not None and self.bounds_max is not None:
            if self.nx > 1:
                self.grid_dx = (self.bounds_max[0] - self.bounds_min[0]) / (self.nx - 1)
            else:
                self.grid_dx = 0.0
            if self.ny > 1:
                self.grid_dy = (self.bounds_max[1] - self.bounds_min[1]) / (self.ny - 1)
            else:
                self.grid_dy = 0.0

        base_count = self.slice_count_original or self.slice_count
        if base_count > 1:
            if self.total_height is not None:
                self.slice_dz = float(self.total_height) / (base_count - 1)
            elif self.bounds_min is not None and self.bounds_max is not None:
                if len(self.bounds_max) >= 3:
                    self.slice_dz = (
                        float(self.bounds_max[2] - self.bounds_min[2])
                        / (base_count - 1)
                    )
                else:
                    self.slice_dz = 1.0
        else:
            self.slice_dz = 0.0

        return self

    def get_slice(self, i: int) -> np.ndarray:
        """
        Input: i (int, slice index)
        Output: np.ndarray, shape (ny, nx) SDF slice.
        """
        if self.sdf_stack is None:
            raise ValueError("SDF stack is not loaded. Call load() first.")
        if i < 0 or i >= self.sdf_stack.shape[0]:
            raise IndexError("slice index %s out of range" % i)
        return self.sdf_stack[i]

    def get_slice_plane(self, i: int) -> PlaneLike:
        """
        Input: i (int, slice index)
        Output: Rhino.Geometry.Plane if Rhino is available; otherwise a tuple.
        """
        if self.slice_dz is None:
            raise ValueError("Slice spacing is not available. Call load() first.")

        z0 = 0.0
        if self.bounds_min is not None and len(self.bounds_min) >= 3:
            z0 = float(self.bounds_min[2])
        z = z0 + (i * self.slice_dz)

        try:
            import Rhino.Geometry as _rg  # type: ignore

            origin = _rg.Point3d(0.0, 0.0, z)
            xaxis = _rg.Vector3d(1.0, 0.0, 0.0)
            yaxis = _rg.Vector3d(0.0, 1.0, 0.0)
            return _rg.Plane(origin, xaxis, yaxis)
        except Exception:
            return ((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    # Operations
    def interpolate_slices(self, i0: int, i1: int, t: float) -> np.ndarray:
        """
        Input: i0 (int), i1 (int), t (float in [0, 1])
        Output: np.ndarray, interpolated SDF slice.
        """
        s0 = self.get_slice(i0)
        s1 = self.get_slice(i1)
        t = float(t)
        return (1.0 - t) * s0 + t * s1

    def build_interpolated_stack(self, target_count: int) -> np.ndarray:
        """
        Input: target_count (int), number of slices after upsampling
        Output: np.ndarray, new stack (target_count, ny, nx)
        """
        if self.sdf_stack is None:
            raise ValueError("SDF stack is not loaded. Call load() first.")
        if target_count <= 0:
            raise ValueError("target_count must be > 0.")

        source = self.sdf_stack
        source_count = source.shape[0]
        if target_count == source_count:
            self.slice_count_current = target_count
            return source

        if target_count == 1:
            new_stack = source[:1].copy()
        else:
            new_stack = np.zeros((target_count, source.shape[1], source.shape[2]))
            for i in range(target_count):
                pos = (i / float(target_count - 1)) * (source_count - 1)
                i0 = int(np.floor(pos))
                i1 = int(np.ceil(pos))
                if i0 == i1:
                    new_stack[i] = source[i0]
                else:
                    t = pos - i0
                    new_stack[i] = (1.0 - t) * source[i0] + t * source[i1]

        self.sdf_stack = new_stack
        self.slice_count_current = target_count

        if target_count > 1:
            if self.total_height is not None:
                self.slice_dz = float(self.total_height) / (target_count - 1)
            elif self.bounds_min is not None and self.bounds_max is not None:
                if len(self.bounds_max) >= 3:
                    self.slice_dz = (
                        float(self.bounds_max[2] - self.bounds_min[2])
                        / (target_count - 1)
                    )
        return new_stack

    def resample_planes_from_curve(
        self, frames: Sequence[PlaneLike], target_count: int
    ) -> List[PlaneLike]:
        """
        Input: frames (Sequence of planes), target_count (int)
        Output: list of planes resampled to target_count.
        """
        if target_count <= 0:
            raise ValueError("target_count must be > 0.")
        if not frames:
            return []
        if len(frames) == target_count:
            return list(frames)
        if target_count == 1:
            return [frames[0]]

        resampled = []
        try:
            import Rhino.Geometry as _rg  # type: ignore

            for i in range(target_count):
                pos = (i / float(target_count - 1)) * (len(frames) - 1)
                i0 = int(np.floor(pos))
                i1 = int(np.ceil(pos))
                if i0 == i1:
                    resampled.append(frames[i0])
                    continue

                t = pos - i0
                f0 = frames[i0]
                f1 = frames[i1]

                origin = (1.0 - t) * f0.Origin + t * f1.Origin
                xaxis = (1.0 - t) * f0.XAxis + t * f1.XAxis
                yaxis = (1.0 - t) * f0.YAxis + t * f1.YAxis
                if not xaxis.Unitize() or not yaxis.Unitize():
                    resampled.append(frames[i0])
                    continue

                zaxis = _rg.Vector3d.CrossProduct(xaxis, yaxis)
                if not zaxis.Unitize():
                    resampled.append(frames[i0])
                    continue
                yaxis = _rg.Vector3d.CrossProduct(zaxis, xaxis)
                yaxis.Unitize()
                resampled.append(_rg.Plane(origin, xaxis, yaxis))
        except Exception:
            for i in range(target_count):
                pos = (i / float(target_count - 1)) * (len(frames) - 1)
                idx = int(round(pos))
                resampled.append(frames[idx])

        return resampled

    def redistance_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Input: slice_2d (np.ndarray, shape (ny, nx))
        Output: np.ndarray, true SDF slice in world units.
        """
        if self.grid_dx is None or self.grid_dy is None:
            raise ValueError("Grid spacing not available. Call load() first.")

        try:
            from scipy.ndimage import distance_transform_edt
        except Exception as exc:
            raise ImportError("scipy is required for re-distance.") from exc

        mask = slice_2d <= 0.0
        dist_in = distance_transform_edt(mask, sampling=(self.grid_dy, self.grid_dx))
        dist_out = distance_transform_edt(~mask, sampling=(self.grid_dy, self.grid_dx))
        return dist_out - dist_in

    def redistance_stack(self, sdf_stack: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Input: sdf_stack (np.ndarray or None)
        Output: np.ndarray, true SDF stack in world units.
        """
        if sdf_stack is None:
            if self.sdf_stack is None:
                raise ValueError("SDF stack is not loaded. Call load() first.")
            sdf_stack = self.sdf_stack

        redist = np.zeros_like(sdf_stack, dtype=float)
        for i in range(sdf_stack.shape[0]):
            redist[i] = self.redistance_slice(sdf_stack[i])
        return redist



    # Curves
    def slice_to_segments(self, slice_2d: np.ndarray) -> List[Segment]:
        """
        Input: slice_2d (np.ndarray, shape (ny, nx))
        Output: list of segments as (p0, p1) in grid coords.
        """
        iso_level = self.iso_level
        ny, nx = slice_2d.shape

        def _interp(p1, p2, v1, v2):
            if abs(v2 - v1) < 1e-12:
                t = 0.5
            else:
                t = (iso_level - v1) / (v2 - v1)
            return p1 + t * (p2 - p1)

        segments: List[Segment] = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v00 = slice_2d[iy, ix]
                v10 = slice_2d[iy, ix + 1]
                v11 = slice_2d[iy + 1, ix + 1]
                v01 = slice_2d[iy + 1, ix]

                case = 0
                if v00 >= iso_level:
                    case |= 1
                if v10 >= iso_level:
                    case |= 2
                if v11 >= iso_level:
                    case |= 4
                if v01 >= iso_level:
                    case |= 8
                if case == 0 or case == 15:
                    continue

                p00 = np.array([ix, iy], dtype=float)
                p10 = np.array([ix + 1, iy], dtype=float)
                p11 = np.array([ix + 1, iy + 1], dtype=float)
                p01 = np.array([ix, iy + 1], dtype=float)

                e0 = _interp(p00, p10, v00, v10)
                e1 = _interp(p10, p11, v10, v11)
                e2 = _interp(p11, p01, v11, v01)
                e3 = _interp(p01, p00, v01, v00)

                if case in (1, 14):
                    segments.append((e3, e0))
                elif case in (2, 13):
                    segments.append((e0, e1))
                elif case in (3, 12):
                    segments.append((e3, e1))
                elif case in (4, 11):
                    segments.append((e1, e2))
                elif case in (5, 10):
                    segments.append((e3, e0))
                    segments.append((e1, e2))
                elif case in (6, 9):
                    segments.append((e0, e2))
                elif case in (7, 8):
                    segments.append((e3, e2))

        return segments

    def segments_to_polylines(self, segments: Sequence[Segment]) -> List[Polyline]:
        """
        Input: segments (Sequence of Segment)
        Output: list of polylines, each as list of 2D points in grid coords.
        """
        if not segments:
            return []

        tol = float(self.tol)

        def _key(pt):
            return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

        seg_map = {}
        for a, b in segments:
            ka = _key(a)
            kb = _key(b)
            seg_map.setdefault(ka, []).append(b)
            seg_map.setdefault(kb, []).append(a)

        polylines: List[Polyline] = []
        visited = set()
        for a, b in segments:
            ka = _key(a)
            kb = _key(b)
            if (ka, kb) in visited or (kb, ka) in visited:
                continue

            poly = [a, b]
            visited.add((ka, kb))

            while True:
                last = poly[-1]
                kl = _key(last)
                neighbors = seg_map.get(kl, [])
                next_pt = None
                for npt in neighbors:
                    kn = _key(npt)
                    if (kl, kn) not in visited and (kn, kl) not in visited:
                        next_pt = npt
                        visited.add((kl, kn))
                        break
                if next_pt is None:
                    break
                poly.append(next_pt)

            while True:
                first = poly[0]
                kf = _key(first)
                neighbors = seg_map.get(kf, [])
                next_pt = None
                for npt in neighbors:
                    kn = _key(npt)
                    if (kn, kf) not in visited and (kf, kn) not in visited:
                        next_pt = npt
                        visited.add((kn, kf))
                        break
                if next_pt is None:
                    break
                poly.insert(0, next_pt)

            if len(poly) >= 2:
                polylines.append(poly)

        return polylines

    def polylines_to_curves(self, polylines: Sequence[Polyline]) -> List["rg.PolylineCurve"]:
        """
        Input: polylines (Sequence of polyline points in grid coords)
        Output: list of Rhino.Geometry.PolylineCurve
        """
        if self.bounds_min is None or self.bounds_max is None:
            raise ValueError("Bounds not available. Call load() first.")
        if self.nx is None or self.ny is None:
            raise ValueError("Grid size not available. Call load() first.")

        try:
            import Rhino.Geometry as _rg  # type: ignore
        except Exception:
            raise ImportError("Rhino.Geometry is required to build curves.")

        xmin, ymin = float(self.bounds_min[0]), float(self.bounds_min[1])
        xmax, ymax = float(self.bounds_max[0]), float(self.bounds_max[1])

        def _to_world(pt):
            x = xmin + (pt[0] / (self.nx - 1)) * (xmax - xmin)
            y = ymin + (pt[1] / (self.ny - 1)) * (ymax - ymin)
            return _rg.Point3d(x, y, 0.0)

        curves: List["rg.PolylineCurve"] = []
        for poly in polylines:
            pts = [_to_world(p) for p in poly]
            curves.append(_rg.PolylineCurve(pts))
        return curves

    def smooth_curves(self, curves: Sequence["rg.Curve"], method: Optional[str] = None) -> List["rg.Curve"]:
        """
        Input: curves (Sequence of Rhino curves), method (str or None)
        Output: list of curves, smoothed if method supports it.
        """
        if method is None:
            method = self.smooth_method
        method = str(method).lower()

        if method in ("none", "off"):
            return list(curves)

        if method == "native":
            try:
                smoothed = []
                for crv in curves:
                    new_crv = crv.DuplicateCurve()
                    new_crv.Smooth(self.smooth_strength, self.smooth_iters)
                    smoothed.append(new_crv)
                return smoothed
            except Exception:
                return list(curves)

        return list(curves)

    # Mesh
    def stack_to_mesh(self, sdf_stack: Optional[np.ndarray] = None) -> Any:
        """
        Input: sdf_stack (np.ndarray or None)
        Output: mesh object (Rhino mesh or equivalent).
        """
        raise NotImplementedError

    def smooth_mesh(self, mesh: Any, method: Optional[str] = None) -> Any:
        """
        Input: mesh (mesh object), method (str or None)
        Output: mesh object after smoothing.
        """
        raise NotImplementedError


# Example usage (GH CPython):
# helper = SdfGhHelper("output/debug_bracing_results.npz", field="result_fields", iso_level=0.0)
# helper.load()
# helper.build_interpolated_stack(target_count=60)
# interp_slice = helper.interpolate_slices(0, 1, 0.5)
# segments = helper.slice_to_segments(interp_slice)
# polylines = helper.segments_to_polylines(segments)
# curves = helper.polylines_to_curves(polylines)


# import importlib
# import debug_sdf_gh
# importlib.reload(debug_sdf_gh)

import ghpythonlib.treehelpers as tr
import numpy as np
import Grasshopper # type: ignore
import Rhino # type: ignore


# inputs
npz_path: Optional[str]
index_list: int

# outputs
a: Any
b: Any

# ----------

def main(npz_path: str, preview_length: Optional[int] = None, max_slices_load: Optional[int] = None):

    helper = SdfGhHelper(npz_path, field="bracing_fields", iso_level=0.0)
    helper.load(max_slices_load)
    shape = np.shape(helper.sdf_stack)

    print(f"slice_number = {shape[0]}, shape_size = {shape[-2:]}")

    # sdf_interpolated = helper.build_interpolated_stack(target_count=500)
    # sdf_redist = helper.redistance_stack()
    # optional: replace current stack
    # helper.sdf_stack = sdf_redist
    if preview_length is not None and preview_length > 0:
        sdf_count = min(preview_length, shape[0])
    else:
        sdf_count = shape[0]
    crv_list = []
    planes = []
    for i in range(sdf_count):

        sdf = helper.get_slice(i)
        segments = helper.slice_to_segments(sdf)
        polylines = helper.segments_to_polylines(segments)
        curves = helper.polylines_to_curves(polylines)

        planes.append(helper.get_slice_plane(i))
        crv_list.append(curves)
    # print(f"crv_list  has {len(crv_list)} \n")
    print(crv_list[0])

    return crv_list, planes



crvs,planes=main(npz_path,preview_length=10)

# print crvs data structure layers
print(f"crvs type: {type(crvs)}")
print(f"crvs length: {len(crvs)}")
print(f"crvs[0] type: {type(crvs[0])}")
print(f"crvs[0] length: {len(crvs[0])}")
print(f"crvs[0][0] type: {type(crvs[0][0])}")


# crv_tree = Grasshopper.DataTree[Rhino.Geometry.Curve]()
# for i in range(len(crvs)):
#     path = Grasshopper.Kernel.Data.GH_Path(i)
#     for crv in crvs[i]:
#         crv_tree.Add(crv, path)

a = tr.list_to_tree(crvs,True)
# a = crv_tree
b = tr.list_to_tree(planes)
