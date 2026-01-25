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
        self.guiding_curve = None
        self.guiding_planes = None

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

    # Mesh guidance
    def load_crv(
        self,
        curve: "rg.Curve",
        frames: Optional[Sequence["rg.Plane"]] = None,
        frame_count: Optional[int] = None,
    ) -> "SdfGhHelper":
        """
        Input: curve (Rhino.Geometry.Curve), frames (optional list of planes),
               frame_count (optional int)
        Output: SdfGhHelper (self), with guiding_curve and guiding_planes populated.
        """
        try:
            import Rhino.Geometry as _rg  # type: ignore
        except Exception as exc:
            raise ImportError("Rhino.Geometry is required for load_crv.") from exc

        if curve is None:
            raise ValueError("curve must be a Rhino.Geometry.Curve.")

        if frame_count is None:
            if self.slice_count_current is not None:
                frame_count = int(self.slice_count_current)
            elif self.slice_count is not None:
                frame_count = int(self.slice_count)
            else:
                raise ValueError("Slice count is not available. Call load() first.")

        if frame_count <= 0:
            raise ValueError("frame_count must be > 0.")

        self.guiding_curve = curve
        if frames:
            self.guiding_planes = self.resample_planes_from_curve(frames, frame_count)
        else:
            self.guiding_planes = self.build_guiding_planes(curve, frame_count)

        if not self.guiding_planes:
            raise ValueError("Failed to build guiding planes from curve.")

        return self

    def build_guiding_planes(
        self, curve: "rg.Curve", count: int
    ) -> List["rg.Plane"]:
        """
        Input: curve (Rhino.Geometry.Curve), count (int)
        Output: list of Rhino.Geometry.Plane, evenly distributed along curve length.
        """
        if count <= 0:
            raise ValueError("count must be > 0.")

        try:
            import Rhino.Geometry as _rg  # type: ignore
        except Exception as exc:
            raise ImportError("Rhino.Geometry is required for build_guiding_planes.") from exc

        if count == 1:
            t = curve.Domain.Mid
            ok, plane = curve.PerpendicularFrameAt(t)
            return [plane] if ok else []

        params = curve.DivideByCount(count - 1, True)
        if not params:
            d0 = curve.Domain.T0
            d1 = curve.Domain.T1
            params = [d0 + (i / float(count - 1)) * (d1 - d0) for i in range(count)]

        planes: List[_rg.Plane] = []
        for t in params:
            ok, plane = curve.PerpendicularFrameAt(t)
            if ok:
                planes.append(plane)

        return planes

    def _interp_plane(self, p0: "rg.Plane", p1: "rg.Plane", t: float) -> "rg.Plane":
        """
        Input: p0 (Plane), p1 (Plane), t (float in [0, 1])
        Output: interpolated Plane.
        """
        import Rhino.Geometry as _rg  # type: ignore

        origin = (1.0 - t) * p0.Origin + t * p1.Origin
        xaxis = (1.0 - t) * p0.XAxis + t * p1.XAxis
        yaxis = (1.0 - t) * p0.YAxis + t * p1.YAxis
        if not xaxis.Unitize() or not yaxis.Unitize():
            return p0

        zaxis = _rg.Vector3d.CrossProduct(xaxis, yaxis)
        if not zaxis.Unitize():
            return p0
        yaxis = _rg.Vector3d.CrossProduct(zaxis, xaxis)
        yaxis.Unitize()
        return _rg.Plane(origin, xaxis, yaxis)

    def _source_plane_at_z(self, z: float) -> "rg.Plane":
        """
        Input: z (float)
        Output: Plane centered on SDF bounds at height z, aligned to WorldXY.
        """
        import Rhino.Geometry as _rg  # type: ignore

        if self.bounds_min is None or self.bounds_max is None:
            raise ValueError("Bounds not available. Call load() first.")

        cx = 0.5 * (self.bounds_min[0] + self.bounds_max[0])
        cy = 0.5 * (self.bounds_min[1] + self.bounds_max[1])
        origin = _rg.Point3d(float(cx), float(cy), float(z))
        return _rg.Plane(origin, _rg.Vector3d.XAxis, _rg.Vector3d.YAxis)

    def _plane_for_z(self, z: float) -> "rg.Plane":
        """
        Input: z (float)
        Output: guiding Plane interpolated by z position.
        """
        if not self.guiding_planes:
            raise ValueError("Guiding planes are not set. Call load_crv() first.")
        if self.bounds_min is None or self.bounds_max is None:
            raise ValueError("Bounds not available. Call load() first.")

        zmin = float(self.bounds_min[2]) if len(self.bounds_min) >= 3 else 0.0
        zmax = float(self.bounds_max[2]) if len(self.bounds_max) >= 3 else zmin
        count = len(self.guiding_planes)
        if count == 1:
            return self.guiding_planes[0]

        if zmax <= zmin + 1e-9:
            if self.slice_dz is not None and self.slice_dz > 0.0:
                zmax = zmin + self.slice_dz * (count - 1)
            else:
                zmax = zmin + max(count - 1, 1)
        if zmax <= zmin + 1e-9:
            return self.guiding_planes[0]

        pos = (z - zmin) / (zmax - zmin)
        pos = min(max(pos, 0.0), 1.0) * (count - 1)
        i0 = int(np.floor(pos))
        i1 = int(np.ceil(pos))
        if i0 == i1:
            return self.guiding_planes[i0]
        t = pos - i0
        return self._interp_plane(self.guiding_planes[i0], self.guiding_planes[i1], t)

    def _warp_vertices_to_guiding_planes(self, vertices: np.ndarray) -> np.ndarray:
        """
        Input: vertices (np.ndarray, shape (N, 3) in world coords)
        Output: np.ndarray, warped vertices along guiding planes.
        """
        import Rhino.Geometry as _rg  # type: ignore

        if not self.guiding_planes:
            return vertices

        warped = np.zeros_like(vertices, dtype=float)
        for i, v in enumerate(vertices):
            z = float(v[2])
            source_plane = self._source_plane_at_z(z)
            target_plane = self._plane_for_z(z)
            xform = _rg.Transform.PlaneToPlane(source_plane, target_plane)
            pt = _rg.Point3d(float(v[0]), float(v[1]), float(v[2]))
            pt.Transform(xform)
            warped[i, 0] = pt.X
            warped[i, 1] = pt.Y
            warped[i, 2] = pt.Z
        return warped

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
        if sdf_stack is None:
            if self.sdf_stack is None:
                raise ValueError("SDF stack is not loaded. Call load() first.")
            sdf_stack = self.sdf_stack

        if self.bounds_min is None or self.bounds_max is None:
            raise ValueError("Bounds not available. Call load() first.")

        try:
            from skimage import measure
        except Exception as exc:
            raise ImportError("scikit-image is required for marching cubes.") from exc

        nx = sdf_stack.shape[2]
        ny = sdf_stack.shape[1]
        nz = sdf_stack.shape[0]

        if self.grid_dx is None:
            self.grid_dx = (
                (self.bounds_max[0] - self.bounds_min[0]) / (nx - 1) if nx > 1 else 0.0
            )
        if self.grid_dy is None:
            self.grid_dy = (
                (self.bounds_max[1] - self.bounds_min[1]) / (ny - 1) if ny > 1 else 0.0
            )

        dz = self.slice_dz
        if dz is None or dz <= 0.0:
            if self.total_height is not None and nz > 1:
                dz = float(self.total_height) / (nz - 1)
            elif len(self.bounds_max) >= 3 and nz > 1:
                dz = float(self.bounds_max[2] - self.bounds_min[2]) / (nz - 1)
            else:
                dz = 1.0

        origin = (
            float(self.bounds_min[0]),
            float(self.bounds_min[1]),
            float(self.bounds_min[2]) if len(self.bounds_min) >= 3 else 0.0,
        )

        verts, faces, _normals, _values = measure.marching_cubes(
            sdf_stack, level=self.iso_level, spacing=(dz, self.grid_dy, self.grid_dx)
        )

        verts_xyz = np.zeros_like(verts)
        verts_xyz[:, 0] = verts[:, 2] + origin[0]
        verts_xyz[:, 1] = verts[:, 1] + origin[1]
        verts_xyz[:, 2] = verts[:, 0] + origin[2]

        if self.guiding_planes:
            verts_xyz = self._warp_vertices_to_guiding_planes(verts_xyz)

        try:
            import Rhino.Geometry as _rg  # type: ignore
        except Exception as exc:
            raise ImportError("Rhino.Geometry is required to build a mesh.") from exc

        mesh = _rg.Mesh()
        for v in verts_xyz:
            mesh.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
        for f in faces:
            mesh.Faces.AddFace(int(f[0]), int(f[1]), int(f[2]))
        mesh.Normals.ComputeNormals()
        mesh.Compact()
        return mesh

    def smooth_mesh(self, mesh: Any, method: Optional[str] = None) -> Any:
        """
        Input: mesh (mesh object), method (str or None)
        Output: mesh object after smoothing.
        """
        # TODO: add laplacian/taubin smoothing with Rhino or open3d
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
import Rhino.Geometry as rg # type: ignore

# inputs
npz_path: Optional[str]
index_list: int
GuideCurve: Optional[rg.Curve]

# outputs
a: Any
b: Any

# ----------

def main(npz_path: str, 
         preview_length: Optional[int] = None, 
         max_slices_load: Optional[int] = None,
         guide_curve: Optional[Rhino.Geometry.Curve] = None):

    helper = SdfGhHelper(npz_path, field="bracing_fields", iso_level=0.0)
    helper.load(max_slices_load)
    helper.load_crv(guide_curve) if guide_curve is not None else None
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
        target_pln = helper.guiding_planes[i]
        target_pln = rg.Plane(target_pln.Origin, target_pln.YAxis, target_pln.XAxis)
        xform = rg.Transform.PlaneToPlane(
            rg.Plane.WorldXY, target_pln
        )
        segments = helper.slice_to_segments(sdf)
        polylines = helper.segments_to_polylines(segments)
        curves = helper.polylines_to_curves(polylines)
        oriented_contours = []
        for crv in curves:
            c = crv.DuplicateCurve()
            c.Transform(xform)
            oriented_contours.append(c)

        planes.append(helper.get_slice_plane(i))
        crv_list.append(oriented_contours)
    # print(f"crv_list  has {len(crv_list)} \n")
    mesh = helper.stack_to_mesh(helper.sdf_stack)
    print(crv_list[0])

    return crv_list, planes, mesh


# Mesh Logic needs update here, use morphed along the cuvre

# crvs,planes,mesh=main(npz_path,
#                  preview_length=None,
#                  guide_curve=GuideCurve)

# print crvs data structure layers
# print(f"crvs type: {type(crvs)}")
# print(f"crvs length: {len(crvs)}")
# print(f"crvs[0] type: {type(crvs[0])}")
# print(f"crvs[0] length: {len(crvs[0])}")
# print(f"crvs[0][0] type: {type(crvs[0][0])}")


# crv_tree = Grasshopper.DataTree[Rhino.Geometry.Curve]()
# for i in range(len(crvs)):
#     path = Grasshopper.Kernel.Data.GH_Path(i)
#     for crv in crvs[i]:
#         crv_tree.Add(crv, path)

# a = tr.list_to_tree(crvs,True)
# # a = crv_tree
# b = mesh

# Mesh smoke test + hints (GH CPython):
helper = SdfGhHelper(npz_path, field="bracing_fields", iso_level=0.0)
helper.load(max_slices=60)
helper.load_crv(GuideCurve)  # evenly distributed frames by slice count
sdf_stack = helper.build_interpolated_stack(target_count=120)
sdf_stack = helper.redistance_stack(sdf_stack)
mesh = helper.stack_to_mesh(sdf_stack) 
a = mesh
# Hints:
# - use planar curves for stable PerpendicularFrameAt
# - if you already have frames, pass them into load_crv(frames=frames)
# - mesh generation requires scikit-image for marching cubes
