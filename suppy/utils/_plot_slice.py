"""Slice visualization of treatment-plan volumes (image, structures, dose)."""

from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

_PLANES = {"axial": 0, "coronal": 1, "sagittal": 2}

# (horizontal, vertical, slice) coordinate axis for each plane, using the
# coordinate order (x, y, z) -> (0, 1, 2)
_PLANE_AXES = {0: (0, 1, 2), 1: (2, 0, 1), 2: (2, 1, 0)}


def _to_numpy(arr):
    """Return ``arr`` as a numpy array (moving cupy arrays to the host)."""
    if arr is None:
        return None
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr)


def _abs_max(arr: np.ndarray) -> float:
    """Return the maximum absolute finite value of ``arr`` (0 if there is
    none).
    """
    finite = np.abs(arr[np.isfinite(arr)])
    return float(finite.max()) if finite.size > 0 else 0.0


def matrad_to_zyx(values: npt.ArrayLike, dims: Sequence[int]) -> np.ndarray:
    """
    Convert a matRad vector or cube into a ``(z, y, x)`` ordered cube.

    matRad stores cubes with dimensions ``(ny, nx, nz)`` and indexes voxels
    by column-major (Fortran) linear indices. This function reshapes a
    flat vector (e.g. a dose vector ``dij @ w``) or reorders a matRad cube
    into the ``(z, y, x)`` layout expected by :func:`plot_slice`.

    Parameters
    ----------
    values : npt.ArrayLike
        Flat vector of length ``prod(dims)`` or a cube of shape ``dims``.
    dims : Sequence[int]
        matRad cube dimensions ``(ny, nx, nz)``.

    Returns
    -------
    np.ndarray
        Cube of shape ``(nz, ny, nx)``.
    """
    values = _to_numpy(values)
    dims = tuple(int(d) for d in np.ravel(dims))
    if values.ndim == 1:
        values = values.reshape(dims, order="F")
    elif values.shape != dims:
        raise ValueError(f"Cube shape {values.shape} does not match dims {dims}")
    return np.transpose(values, (2, 0, 1))


def matrad_indices_to_mask(
    idxs: npt.ArrayLike, dims: Sequence[int], one_based: bool = False
) -> np.ndarray:
    """
    Build a ``(z, y, x)`` ordered boolean mask from matRad voxel indices.

    Parameters
    ----------
    idxs : npt.ArrayLike
        Column-major linear voxel indices of the structure (as stored in
        ``cst{i, 4}``).
    dims : Sequence[int]
        matRad cube dimensions ``(ny, nx, nz)``.
    one_based : bool, optional
        Set to True if ``idxs`` are MATLAB (1-based) indices, by default
        False.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(nz, ny, nx)``.
    """
    idxs = np.ravel(_to_numpy(idxs)).astype(np.int64)
    if one_based:
        idxs = idxs - 1
    mask = np.zeros(int(np.prod(dims)), dtype=bool)
    mask[idxs] = True
    return matrad_to_zyx(mask, dims)


class _SlicePlotter:
    """Preprocessed volumes and display settings shared by the slice plots.

    See :func:`plot_slice` for a description of the parameters.
    """

    def __init__(
        self,
        image: Optional[npt.ArrayLike] = None,
        structures: Optional[Dict[str, npt.ArrayLike]] = None,
        overlay: Optional[npt.ArrayLike] = None,
        *,
        resolution: Sequence[float] = (1.0, 1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        plane: Union[str, int] = "axial",
        target: Optional[str] = None,
        structure_colors: Optional[Dict[str, object]] = None,
        contour_line_width: float = 1.0,
        show_legend: bool = False,
        image_window: Optional[Tuple[float, float]] = None,
        window_mode: str = "minmax",
        image_cmap: str = "bone",
        overlay_alpha: Optional[float] = None,
        overlay_rel_threshold: float = 0.01,
        overlay_window: Optional[Tuple[float, float]] = None,
        overlay_cmap: Optional[str] = None,
        overlay_label: str = "",
        use_global_max: bool = False,
        scale_bar: Optional[float] = 50.0,
        zoom: Optional[Union[str, Sequence[str], Tuple[Tuple[float, float], ...]]] = None,
        zoom_margin: float = 20.0,
        matrad_style: bool = False,
    ):
        self.image = _to_numpy(image)
        self.overlay = _to_numpy(overlay)
        self.structures = {name: _to_numpy(mask) for name, mask in (structures or {}).items()}

        cubes = [c for c in [self.image, self.overlay, *self.structures.values()] if c is not None]
        if len(cubes) == 0:
            raise ValueError("Nothing to visualize!")
        self.shape = cubes[0].shape
        if len(self.shape) != 3 or any(c.shape != self.shape for c in cubes):
            raise ValueError("All volumes must be 3D and share the same (z, y, x) shape")

        self.plane = _PLANES.get(plane, plane)
        if self.plane not in (0, 1, 2):
            raise ValueError("Invalid plane")

        # world coordinates of the voxel centers in (x, y, z) order
        n_xyz = (self.shape[2], self.shape[1], self.shape[0])
        coords = [origin[i] + resolution[i] * np.arange(n_xyz[i]) for i in range(3)]
        h_axis, v_axis, s_axis = _PLANE_AXES[self.plane]
        h_coords, v_coords, self.z_plot = coords[h_axis], coords[v_axis], coords[s_axis]
        self.h_coords, self.v_coords = h_coords, v_coords

        # default slice: center of mass of the target or the middle slice
        if target is not None and self.structures[target].any():
            com = np.argwhere(self.structures[target]).mean(axis=0)
            self.default_slice = int(np.round(com[self.plane]))
        else:
            self.default_slice = int(self.shape[self.plane] // 2)

        # full extent (voxel edges) along the horizontal and vertical plot axes
        h_half, v_half = resolution[h_axis] / 2, resolution[v_axis] / 2
        self.h_full = (h_coords[0] - h_half, h_coords[-1] + h_half)
        self.v_full = (v_coords[0] - v_half, v_coords[-1] + v_half)

        # visible region
        if zoom is None:
            self.h_lim, self.v_lim = self.h_full, self.v_full
        elif isinstance(zoom, str) or isinstance(zoom[0], str):
            names = [zoom] if isinstance(zoom, str) else list(zoom)
            voxels = np.argwhere(np.any([self.structures[name] for name in names], axis=0))
            if voxels.size == 0:
                raise ValueError(f"Cannot zoom to empty structure(s) {names}")
            # cube axes are (z, y, x), i.e. world axis i is cube axis 2 - i
            h_idx, v_idx = voxels[:, 2 - h_axis], voxels[:, 2 - v_axis]
            self.h_lim = (
                max(h_coords[h_idx.min()] - h_half - zoom_margin, self.h_full[0]),
                min(h_coords[h_idx.max()] + h_half + zoom_margin, self.h_full[1]),
            )
            self.v_lim = (
                max(v_coords[v_idx.min()] - v_half - zoom_margin, self.v_full[0]),
                min(v_coords[v_idx.max()] + v_half + zoom_margin, self.v_full[1]),
            )
        else:
            self.h_lim, self.v_lim = tuple(sorted(zoom[0])), tuple(sorted(zoom[1]))

        # image display window
        if self.image is not None:
            if image_window is None:
                image_window = (float(np.min(self.image)), float(np.max(self.image)))
            elif window_mode == "centerwidth":
                center, width = image_window
                image_window = (center - width / 2, center + width / 2)
            elif window_mode != "minmax":
                raise ValueError(f"Invalid window_mode: {window_mode}")
            if image_window[0] > image_window[1]:
                raise ValueError("Invalid image_window: min must be less than max")
            if matrad_style and image_window[1] - image_window[0] <= 0:
                image_window = (-1000.0, 2000.0)
        self.image_window = image_window

        # overlays with negative values (e.g. difference maps) are shown on a
        # symmetric colormap around zero
        self.signed_overlay = self.overlay is not None and bool(np.nanmin(self.overlay) < 0)
        if overlay_cmap is None:
            overlay_cmap = "RdBu_r" if self.signed_overlay else "jet"
        if overlay_alpha is None:
            overlay_alpha = 0.6 if matrad_style else 0.5

        # matRad display: 64-color maps, threshold and default window from the
        # whole overlay cube, overlay values outside the window transparent
        if matrad_style:
            image_cmap = plt.get_cmap(image_cmap).resampled(64)
            overlay_cmap = plt.get_cmap(overlay_cmap).resampled(64)
            use_global_max = True
            if self.overlay is not None and overlay_window is None:
                # 1.001 as in the matRad GUI, so the maximum stays visible
                o_max = 1.001 * _abs_max(self.overlay)
                o_min = -o_max if self.signed_overlay else float(np.nanmin(self.overlay))
                overlay_window = (o_min, o_max)
        self.matrad_style = matrad_style

        self.global_overlay_max = (
            _abs_max(self.overlay) if self.overlay is not None and use_global_max else None
        )

        color_cycle = cycle(plt.get_cmap("tab10").colors)
        self.structure_colors = {
            name: (structure_colors or {}).get(name, next(color_cycle)) for name in self.structures
        }

        self.contour_line_width = contour_line_width
        self.show_legend = show_legend
        self.image_cmap = image_cmap
        self.overlay_alpha = overlay_alpha
        self.overlay_rel_threshold = overlay_rel_threshold
        self.overlay_window = overlay_window
        self.overlay_cmap = overlay_cmap
        self.overlay_label = overlay_label
        self.scale_bar = scale_bar

    def _get_2d(self, cube: np.ndarray, slice_idx: int) -> np.ndarray:
        slice_2d = np.take(cube, slice_idx, axis=self.plane)
        return slice_2d.T if self.plane != 0 else slice_2d

    def draw(self, ax: plt.Axes, slice_idx: int, limits: Optional[tuple] = None):
        """Draw a slice into ``ax`` and return the overlay image (or None).

        ``limits`` ``((h_min, h_max), (v_min, v_max))`` overrides the
        visible region.
        """
        h_lim, v_lim = limits if limits is not None else (self.h_lim, self.v_lim)
        h_width, v_width = h_lim[1] - h_lim[0], v_lim[1] - v_lim[0]
        # row 0 at the top: the vertical axis points downwards
        extent = (self.h_full[0], self.h_full[1], self.v_full[1], self.v_full[0])

        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False,
        )

        if self.image is not None:
            ax.imshow(
                self._get_2d(self.image, slice_idx),
                cmap=self.image_cmap,
                vmin=self.image_window[0],
                vmax=self.image_window[1],
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )

        for name, mask in self.structures.items():
            mask_2d = self._get_2d(mask, slice_idx).astype(float)
            if mask_2d.any():
                ax.contour(
                    self.h_coords,
                    self.v_coords,
                    mask_2d,
                    levels=[0.5],
                    colors=[self.structure_colors[name]],
                    linewidths=self.contour_line_width,
                )

        im = None
        if self.overlay is not None:
            overlay_2d = self._get_2d(self.overlay, slice_idx)
            current_max = (
                self.global_overlay_max
                if self.global_overlay_max is not None
                else _abs_max(overlay_2d)
            )
            if self.overlay_window is not None:
                vmin, vmax = self.overlay_window
            else:
                vmin, vmax = (-current_max if self.signed_overlay else 0), current_max
            # NaN and values with |v| below the threshold are transparent
            values = np.nan_to_num(overlay_2d)
            visible = np.abs(values) > self.overlay_rel_threshold * current_max
            if self.matrad_style:
                visible &= (values > vmin) & (values < vmax)
            im = ax.imshow(
                overlay_2d,
                cmap=self.overlay_cmap,
                alpha=self.overlay_alpha * visible,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                interpolation="nearest",
            )

        if self.show_legend and len(self.structures) > 0:
            handles = [
                Line2D([], [], color=c, lw=self.contour_line_width)
                for c in self.structure_colors.values()
            ]
            ax.legend(handles, list(self.structure_colors), loc="upper right", fontsize=8)

        if self.scale_bar is not None:
            x0 = h_lim[0] + 0.05 * h_width
            y0 = v_lim[1] - 0.05 * v_width
            ax.plot([x0, x0 + self.scale_bar], [y0, y0], "w-", linewidth=3)
            ax.text(
                x0 + self.scale_bar / 2,
                y0 - 0.03 * v_width,
                f"{self.scale_bar:g} mm",
                color="w",
                ha="center",
                fontsize=10,
            )

        ax.set_title(f"Slice ID = {slice_idx} at {self.z_plot[slice_idx]:.1f} mm")
        ax.set_xlim(h_lim)
        ax.set_ylim(v_lim[1], v_lim[0])  # inverted vertical axis
        return im


_PLOT_KWARGS_DOC = """resolution : Sequence[float], optional
        Voxel size ``(dx, dy, dz)`` in mm, by default ``(1, 1, 1)``.
    origin : Sequence[float], optional
        World coordinate ``(x, y, z)`` of the first voxel in mm, by default
        ``(0, 0, 0)``.
    plane : str or int, optional
        ``"axial"`` (0), ``"coronal"`` (1) or ``"sagittal"`` (2), by default
        ``"axial"``.
    target : str, optional
        Name of the structure used to choose the default slice.
    structure_colors : Dict[str, object], optional
        Matplotlib colors per structure name; missing entries are taken
        from the ``tab10`` cycle.
    contour_line_width : float, optional
        Line width of the structure contours, by default 1.0.
    show_legend : bool, optional
        Add a legend with the structure names, by default False.
    image_window : Tuple[float, float], optional
        Display window of the image, interpreted according to
        ``window_mode``. Defaults to the image min/max.
    window_mode : str, optional
        ``"minmax"`` interprets ``image_window`` as ``(min, max)``,
        ``"centerwidth"`` as ``(center, width)``, by default ``"minmax"``.
    image_cmap : str, optional
        Colormap of the image, by default ``"bone"``.
    overlay_alpha : float, optional
        Opacity of the overlay, by default 0.5 (0.6 with ``matrad_style``).
    overlay_rel_threshold : float, optional
        Overlay voxels with ``|value|`` below this fraction of the maximum
        ``|value|`` (and NaN voxels) are transparent, by default 0.01.
    overlay_window : Tuple[float, float], optional
        ``(vmin, vmax)`` of the overlay colormap. Defaults to ``(0, max)``,
        or ``(-max, max)`` if the overlay contains negative values (e.g.
        difference maps), with ``max`` the maximum ``|value|``.
    overlay_cmap : str, optional
        Colormap of the overlay. Defaults to ``"jet"``, or ``"RdBu_r"`` if
        the overlay contains negative values.
    overlay_label : str, optional
        Colorbar label of the overlay (e.g. ``"Gy"``), by default ``""``.
    use_global_max : bool, optional
        Scale the overlay by the maximum of the whole cube instead of the
        maximum of each slice, by default False.
    scale_bar : float, optional
        Length of the scale bar in mm, None to disable, by default 50.
    zoom : str, Sequence[str] or Tuple[Tuple[float, float], ...], optional
        Region to display. A structure name (or list of names) zooms to the
        bounding box of these structures over the whole volume (identical
        for all slices) enlarged by ``zoom_margin``. Alternatively, explicit
        limits ``((h_min, h_max), (v_min, v_max))`` in mm along the
        horizontal and vertical plot axes. Defaults to the full image.
    zoom_margin : float, optional
        Margin in mm around the structures when zooming to structures, by
        default 20.
    matrad_style : bool, optional
        Mimic the matRad slice display, by default False: image and overlay
        colormaps with 64 colors, overlay threshold relative to the maximum
        of the whole cube (``use_global_max``), default overlay window
        ``(min, 1.001 max)`` of the whole cube (symmetric for signed
        overlays), overlay values outside the window transparent instead of
        saturated, overlay opacity 0.6 and image window ``(-1000, 2000)`` if
        the image is constant.
    """


def plot_slice(
    image: Optional[npt.ArrayLike] = None,
    structures: Optional[Dict[str, npt.ArrayLike]] = None,
    overlay: Optional[npt.ArrayLike] = None,
    view_slice: Optional[Union[int, Sequence[int]]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
    **kwargs,
) -> List[plt.Axes]:
    """
    Plot one or multiple slices of an image with structures and an overlay.

    All volumes must share the same grid and be ordered ``(z, y, x)``
    (SimpleITK convention). Use :func:`matrad_to_zyx` and
    :func:`matrad_indices_to_mask` to convert matRad data.

    Parameters
    ----------
    image : npt.ArrayLike, optional
        Background image cube (e.g. CT in HU).
    structures : Dict[str, npt.ArrayLike], optional
        Mapping of structure name to a boolean mask cube; drawn as
        contours.
    overlay : npt.ArrayLike, optional
        Overlay cube (e.g. dose) drawn with a colormap on top.
    view_slice : int or Sequence[int], optional
        Slice indices along the chosen plane. Defaults to the slice through
        the center of mass of ``target`` or, if not given, the middle slice.
    ax : plt.Axes, optional
        Existing axes to draw on; only a single slice is supported then.
    save_filename : str, optional
        Save the figure to this file (only for internally created figures).
    show_plot : bool, optional
        Call ``plt.show()`` (only for internally created figures), by
        default True.
    **kwargs
        Display options, see Other Parameters.

    Returns
    -------
    List[plt.Axes]
        The axes the slices were drawn on.

    Other Parameters
    ----------------
    """
    plotter = _SlicePlotter(image, structures, overlay, **kwargs)

    if view_slice is None:
        view_slice = [plotter.default_slice]
    elif isinstance(view_slice, (int, np.integer)):
        view_slice = [int(view_slice)]

    if ax is not None:
        if len(view_slice) > 1:
            raise ValueError("External axes only supports single slice plotting")
        axes = [ax]
    else:
        num_slices = len(view_slice)
        cols = int(np.ceil(np.sqrt(num_slices)))
        rows = int(np.ceil(num_slices / cols))
        _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = list(axes.flatten())
        for unused_ax in axes[num_slices:]:
            unused_ax.set_axis_off()

    for current_ax, slice_idx in zip(axes, view_slice):
        im = plotter.draw(current_ax, slice_idx)
        if im is not None:
            current_ax.figure.colorbar(im, ax=current_ax, label=plotter.overlay_label)

    if ax is None:
        plt.tight_layout()
        if save_filename is not None:
            plt.savefig(save_filename)
        if show_plot:
            plt.show()

    return axes


def scroll_slices(
    image: Optional[npt.ArrayLike] = None,
    structures: Optional[Dict[str, npt.ArrayLike]] = None,
    overlay: Optional[npt.ArrayLike] = None,
    start_slice: Optional[int] = None,
    *,
    figsize: Tuple[float, float] = (6.0, 6.5),
    show_plot: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes, Slider]:
    """
    Interactively scroll through the slices of a volume.

    Slices are changed with the slider, the mouse wheel or the up/down
    arrow keys (page up/down jump 10 slices). Zooming or panning with the
    toolbar is kept while scrolling. Requires an interactive matplotlib
    backend, in Jupyter e.g. ``%matplotlib widget`` (needs ``ipympl``).
    Keep a reference to the returned slider, otherwise it stops reacting.

    Parameters
    ----------
    image : npt.ArrayLike, optional
        Background image cube (e.g. CT in HU).
    structures : Dict[str, npt.ArrayLike], optional
        Mapping of structure name to a boolean mask cube; drawn as
        contours.
    overlay : npt.ArrayLike, optional
        Overlay cube (e.g. dose) drawn with a colormap on top.
    start_slice : int, optional
        Initially shown slice. Defaults to the slice through the center of
        mass of ``target`` or, if not given, the middle slice.
    figsize : Tuple[float, float], optional
        Figure size in inches, by default ``(6, 6.5)``.
    show_plot : bool, optional
        Call ``plt.show()``, by default True.
    **kwargs
        Display options, see Other Parameters.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes, Slider]
        The figure, the image axes and the slice slider.

    Other Parameters
    ----------------
    """
    plotter = _SlicePlotter(image, structures, overlay, **kwargs)
    n_slices = plotter.shape[plotter.plane]
    start_slice = plotter.default_slice if start_slice is None else int(start_slice)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.12)
    slider_ax = fig.add_axes([0.15, 0.03, 0.65, 0.03])
    slider = Slider(
        slider_ax, "Slice", 0, n_slices - 1, valinit=start_slice, valstep=1, valfmt="%d"
    )

    im = plotter.draw(ax, start_slice)
    cbar = fig.colorbar(im, ax=ax, label=plotter.overlay_label) if im is not None else None

    def _update(val):
        # keep the currently visible region (e.g. after toolbar zoom)
        limits = (tuple(ax.get_xlim()), tuple(sorted(ax.get_ylim())))
        ax.clear()
        im = plotter.draw(ax, int(val), limits)
        if cbar is not None:
            cbar.update_normal(im)
        fig.canvas.draw_idle()

    def _step(step):
        slider.set_val(int(np.clip(slider.val + step, 0, n_slices - 1)))

    def _on_scroll(event):
        _step(1 if event.button == "up" else -1)

    def _on_key(event):
        steps = {"up": 1, "down": -1, "pageup": 10, "pagedown": -10}
        if event.key in steps:
            _step(steps[event.key])

    slider.on_changed(_update)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    if show_plot:
        plt.show()

    return fig, ax, slider


plot_slice.__doc__ += _PLOT_KWARGS_DOC
scroll_slices.__doc__ += _PLOT_KWARGS_DOC
