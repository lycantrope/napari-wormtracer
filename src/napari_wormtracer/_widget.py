import functools
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from scipy import interpolate

if TYPE_CHECKING:
    import napari


def get_barcode() -> str:
    # Update barcode every 5 seconds.
    millisecond = int(datetime.now().timestamp()) // 5 * 1000
    return hex(millisecond)[2:]


class WormTracerUI(QWidget):
    def __init__(
        self, viewer: "napari.viewer.Viewer", parent=None
    ):  # type-hint is required
        super().__init__(parent)

        show_info("WormTracer GUI was loaded")

        self._viewer: napari.Viewer = viewer
        # QPushButton (name, callback)
        btns = [
            ("Load Image", self._load_image),
            ("Load Centerline", self._load_centerline),
            ("Prev (-1)", functools.partial(self._move_frame, step=-1)),
            ("Next (+1)", functools.partial(self._move_frame, step=1)),
            ("Clear", self._reset_centerline),
            ("Reload", self._reload_centerline),
            ("Flip", self._flip),
            ("Register", self._register),
            ("Undo", self._undo),
            ("Save All", self._save_all),
        ]
        ncol = 2
        layout = QGridLayout(self)
        for i, (name, callback) in enumerate(btns):
            btn = QPushButton(self)
            btn.setText(name)
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(108)
            btn.clicked.connect(callback)
            row_idx = i // ncol
            col_idx = i % ncol
            layout.addWidget(
                btn, row_idx, col_idx, Qt.AlignmentFlag.AlignCenter
            )

        group_box = QGroupBox("Output Type", self)
        group_box.setMinimumHeight(60)
        group_box.setMinimumWidth(108)

        vlayout = QVBoxLayout()
        group_box.setLayout(vlayout)

        self.group_buttons = QButtonGroup(self)
        self.group_buttons.setExclusive(True)

        check_btn = QRadioButton(self)
        check_btn.setText("csv")
        self.group_buttons.addButton(check_btn)
        check_btn.setChecked(True)
        vlayout.addWidget(check_btn)

        check_btn = QRadioButton(self)
        check_btn.setText("hdf")
        self.group_buttons.addButton(check_btn)
        vlayout.addWidget(check_btn)

        n_btns = len(btns)
        row_idx = n_btns // ncol
        col_idx = n_btns % ncol

        layout.addWidget(
            group_box, row_idx, col_idx, Qt.AlignmentFlag.AlignCenter
        )

        layout.setSpacing(0)
        layout.setContentsMargins(16, 8, 16, 8)
        self.setLayout(layout)

        self.centerlines = None
        self.is_flip = None
        # Memory the unmodified line. for redo
        self.history = defaultdict(list)
        # src_path
        self.src_path = None

        # layers
        self.shapes_layer = None
        self.nose_tip_layer = None

    def _move_frame(self, step: int):
        z_idx = self._viewer.dims.current_step[0]
        # reset current index
        self._reset_centerline()
        next_step = max(z_idx + step, 0)
        if self.centerlines is not None:
            T = self.centerlines.shape[0]
            next_step = min(next_step, T - 1)
        self._viewer.dims.set_current_step(0, next_step)

    def _save_all(self):
        if self.centerlines is None or self.src_path is None:
            return
        assert self.is_flip is not None, ""

        # Z, Y, X
        x = self.centerlines[:, :, 2]
        y = self.centerlines[:, :, 1]
        # Flip the output
        mask = self.is_flip == 1
        x[mask, :] = x[mask, ::-1].astype("f8")
        y[mask, :] = y[mask, ::-1].astype("f8")

        x_src, y_src = self.src_path
        current_btn = self.group_buttons.checkedButton()
        assert current_btn is not None, "output_type should be checked"
        output_type = current_btn.text()
        parent = x_src.parent
        prefix = os.path.commonprefix([x_src.stem, y_src.stem]).strip("_")
        suffix = get_barcode()
        if output_type == "csv":
            x_dst = parent.joinpath(f"{prefix}_x.{suffix}.csv")
            y_dst = parent.joinpath(f"{prefix}_y.{suffix}.csv")
            np.savetxt(x_dst, x, delimiter=",")
            np.savetxt(y_dst, y, delimiter=",")
            show_info("Modified centerline was saved.")
        else:
            prefix = "_".join(prefix.split("_")[:-1])  # Drop last _xy tag
            dst = parent.joinpath(f"{prefix}_xy.{suffix}.h5")
            with h5py.File(dst, "w") as handler:
                handler.create_dataset("x", data=x)
                handler.create_dataset("y", data=y)

    def _load_centerline(self):
        x_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select .csv file generated by WormTracer (ex: *_x.csv)",
            filter="Csv Files (*.csv);;H5 Files (*.h5);;All Files (*.*)",
        )
        folder = Path(x_path).parent
        name = Path(x_path).name
        if name.endswith(".h5"):
            x_name = name
            y_name = name
        elif "_x" in name:
            x_name = name
            y_name = name.replace("_x", "_y")
        elif "_y" in name:
            y_name = name
            x_name = name.replace("_y", "_x")
        else:
            if self.centerlines is None:
                show_error(
                    f"Select file did not contains proper suffixes (_x or _y): {name}"
                )
            return

        self.src_path = (folder.joinpath(x_name), folder.joinpath(y_name))
        self._reload_centerline()
        self._viewer.dims.set_current_step(0, 0)

    def _reload_centerline(self):
        if self.src_path is None:
            return

        z_idx = 0
        if self.shapes_layer is not None:
            z_idx = self._viewer.dims.current_step[0]
            self._viewer.layers.remove(self.shapes_layer)
            assert self.nose_tip_layer is not None, ""
            self._viewer.layers.remove(self.nose_tip_layer)

        x_src, y_src = self.src_path
        if x_src.name.endswith(".csv"):
            # load x and y
            x = np.loadtxt(x_src, delimiter=",")
            y = np.loadtxt(y_src, delimiter=",")
        elif x_src.name.endswith(".h5"):
            with h5py.File(x_src, "r") as handler:
                x = np.asarray(handler["x"])
                y = np.asarray(handler["y"])

        T, plot_n = x.shape
        z = np.repeat(np.arange(T), plot_n).reshape(T, plot_n)
        self.centerlines = np.stack([z, y, x], axis=-1)  # (1500, 100, 3)
        self.shapes_layer = self._viewer.add_shapes(
            data=list(self.centerlines),
            ndim=3,
            shape_type="path",  # 'path' means polyline in napari
            name="centerline",
            # This parameter is crucial: it tells napari how to group the vertices
            # into separate shapes (one shape per time point in this case)
            face_color="transparent",
            edge_color="yellow",
            edge_width=2,
        )
        self.shapes_layer.editable = True
        # (T, 1, 3) => (T, 4, 3)
        # [4, 3] => [1, 4, 3]
        self.nose_tip_layer = self._viewer.add_points(
            data=[skel[0] for skel in self.centerlines],
            ndim=3,
            name="nose",
            # This parameter is crucial: it tells napari how to group the vertices
            # into separate shapes (one shape per time point in this case)
            face_color="transparent",
            border_color="red",
            size=5,
            border_width=0.15,
        )
        self.nose_tip_layer.editable = False
        # memory whether current centerline was flip or not
        self.is_flip = np.repeat(0, T).astype("u1")
        self._viewer.dims.set_current_step(0, z_idx)

    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select an Image File",
            filter="Image Files (*.png *.tif *.tiff *.jpg *.avi *.mp4,);;All Files (*.*)",
        )
        if not file_path:
            show_warning("Selection aborted!")
            return

        im_layer = self._viewer.open(file_path, stack=True)
        current_index = self._viewer.layers.index(im_layer[0])
        # Move the image to the button
        self._viewer.layers.move(current_index, 0)
        self._viewer.dims.set_current_step(0, 0)

        # layer = self._viewer.add_image(image_data, name=path.split("/")[-1])

    def _register(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, "Some problem occurs"
        assert self.shapes_layer is not None, ""
        assert self.nose_tip_layer is not None, ""
        z_idx = self._viewer.dims.current_step[0]

        data = self.shapes_layer.data
        # Get all shapes associated to current indices
        current_data = [
            (i, d)
            for (i, d) in enumerate(data)
            if d[0, 0] == z_idx and d.shape[0] > 1
        ]

        if not current_data:
            return

        if len(current_data) > 1:
            #  Drop the previous shape
            current_data = [(i, d) for (i, d) in current_data if i != z_idx]

        _, new_shape = min(current_data, key=lambda x: x[0])
        n_pts = new_shape.shape[0]
        arc_length = np.zeros(n_pts)

        square_diff = (new_shape[1:] - new_shape[:-1]) ** 2
        arc_length[1:] = np.sqrt(square_diff.sum(axis=1))
        arc_length = np.cumsum(arc_length)
        # normalized to [0.0, 1.0]
        arc_length /= arc_length.max()
        cs = interpolate.CubicSpline(
            arc_length,
            new_shape,
        )
        plot_n = self.centerlines.shape[1]
        interpolated_data = cs(np.linspace(0, 1.0, plot_n))
        interpolated_data[:, 0] = z_idx

        # Memory the previous centerline for undo.
        self.history[z_idx].append(self.centerlines[z_idx].copy())
        # Assign the interpolated data to centerline
        self.centerlines[z_idx] = interpolated_data
        # reset flip
        self.is_flip[z_idx] = 0
        # Update the centerline using reset_centerline
        self._reset_centerline()

    def _flip(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, "Some problem occurs"
        z_idx = self._viewer.dims.current_step[0]
        self.history[z_idx].append(True)
        self.is_flip[z_idx] = self.is_flip[z_idx] ^ 1
        self._reset_centerline()

    def _reset_centerline(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, ""
        assert self.shapes_layer is not None, ""
        assert self.nose_tip_layer is not None, ""

        data = self.shapes_layer.data
        T = self.centerlines.shape[0]
        if len(data) > T:
            # Remove additional shape
            # Get all z stack indices
            indices = np.array([d[0, 0] for d in data])
            # Get first occurence index.
            z_values, indices = np.unique(indices, return_index=True)
            # sort by z_values
            sorted_indices = indices[np.argsort(z_values)]
            data = [data[i] for i in sorted_indices]

        z_idx = self._viewer.dims.current_step[0]
        skel = self.centerlines[z_idx]

        if self.is_flip[z_idx]:
            skel = skel[::-1, :]

        data[z_idx] = skel
        # we have to assign the data to update the drawing
        self.shapes_layer.data = data

        data = self.nose_tip_layer.data
        data[z_idx] = skel[0]
        # we have to assign the data to update the drawing
        self.nose_tip_layer.data = data

    def _undo(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, ""
        z_idx = self._viewer.dims.current_step[0]
        try:
            prev_skel = self.history[z_idx].pop()
            if isinstance(prev_skel, bool):
                # If previous step is flip, we just revert the flip.
                self.is_flip[z_idx] = self.is_flip[z_idx] ^ 1
            else:
                self.centerlines[z_idx] = prev_skel
        except IndexError:
            # Pop item from empty history list will raise IndexError.
            pass
        self._reset_centerline()
