from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from napari.utils.notifications import show_error, show_info
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import interpolate

if TYPE_CHECKING:
    import napari
    import napari.viewer


def get_barcode() -> str:
    # Update barcode every sec
    millisecond = int(datetime.now().timestamp()) * 1000
    #  136 year as a cycle no repeat
    return hex(millisecond)[5:]


def find_most_commonprefix_name(folder: Path, name: str) -> str:

    # we attempt to find the file by replacing the name. Then, find the commonprefix
    if ("_x" in name) or ("_y" in name):
        if "_x" in name:
            target = name.replace("_x", "_y")
        else:
            target = name.replace("_y", "_x")
        if (folder / target).is_file():
            return target

    _, ext = os.path.splitext(name)
    targets = [dst.name for dst in folder.glob("*" + ext) if name != dst.name]

    # Group the targets file by the length of commonprefix
    grouped_by_prefix = defaultdict(list)

    for p in targets:
        prefix = os.path.commonprefix((p, name))
        grouped_by_prefix[len(prefix)].append(p)

    # Retrieve the longest shared prefix
    longest_prefix = grouped_by_prefix[max(grouped_by_prefix.keys())]

    if len(longest_prefix) == 0:
        raise ValueError(f"Cannot find the corresponding file: {name}")
    elif len(longest_prefix) == 1:
        return longest_prefix[0]

    grouped_by_suffix = defaultdict(list)
    for p in longest_prefix:
        suffix = os.path.commonprefix((p[::-1], name[::-1]))
        grouped_by_suffix[len(suffix)].append(p)

    final_targets = grouped_by_suffix[max(grouped_by_suffix.keys())]
    if len(final_targets) == 0:
        raise ValueError(f"Cannot find the corresponding file: {name}")
    elif len(final_targets) == 1:
        return final_targets[0]

    # If multiple files exist, character distance is used to estimate the closest file.
    def _helper(s1: str, s2: str):
        return sum(abs(ord(c1) - ord(c2)) for c1, c2 in zip(s1, s2))

    return min(final_targets, key=lambda x: _helper(s1=x, s2=name))


class ColorButton(QPushButton):
    """
    Custom Qt Widget to show a chosen color.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = Signal(object)

    def __init__(self, parent, *args, color=None, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._parent = parent

        self._color = None
        self._default = color
        self.pressed.connect(self.onColorPicker)

        # Set the initial/default state.
        self.setColor(self._default)

    @property
    def contrast_color(self):
        if self._color is None:
            return None
        rgba = np.asarray(QColor(self._color).getRgb())

        r, g, b, a = np.nan_to_num(rgba)
        x = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if x > 127.0:
            return "black"
        else:
            return "white"

    def setColor(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit(color)
        if self._color:
            self.setStyleSheet(
                f"background-color: {self._color};color: {self.contrast_color};font: bold;"
            )
        else:
            self.setStyleSheet("font: bold;color: {self.contrast_color};")

    def color(self):
        return self._color

    def onColorPicker(self):
        """
        Show color-picker dialog to select color.

        Qt will use the native dialog by default.

        """
        # This QColorDialog will directly inherit the stylesheet from parent.
        dlg = QColorDialog(parent=self._parent)
        if self._color:
            dlg.setCurrentColor(QColor(self._color))

        if dlg.exec():
            self.setColor(dlg.currentColor().name())

    def mousePressEvent(self, e):
        if e is not None and e.button() == Qt.MouseButton.RightButton:
            self.setColor(self._default)

        return super().mousePressEvent(e)


class GuideRangeGroup(QGroupBox):
    def __init__(self, title, parent):
        super().__init__(title=title, parent=parent)
        self.lower_bound = 0
        self.upper_bound = 100000

        self.start = QSpinBox(self)
        self.start.setSingleStep(1)
        self.start.setMinimum(0)
        self.start.setMaximum(100000)
        self.start.valueChanged.connect(self._update_lower)
        self.l1 = QLabel(parent=self, text="From:")
        self.end = QSpinBox(self)
        self.end.setMinimum(0)
        self.end.setMaximum(self.upper_bound)

        self.end.setSingleStep(1)
        self.end.valueChanged.connect(self._update_upper)

        self.l2 = QLabel(parent=self, text="To:")

        # 3. Connect the signal

        self.label_btn = QPushButton(self)
        self.label_btn.setText("Mark")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignJustify)
        layout.addWidget(self.l1)
        layout.addWidget(self.start)
        layout.addWidget(self.l2)
        layout.addWidget(self.end)
        layout.addWidget(self.label_btn)

    def _update_upper(self, value):
        self.start.setMaximum(max(value - 1, 0))

    def _update_lower(self, value):
        self.end.setMinimum(value + 1)

    def setRange(
        self,
        min_: int,
        max_: int,
        lower_bound: int,
        upper_bound: int,
    ):
        self.l1.setText(f"From (min:{lower_bound}):")
        self.l2.setText(f"To (max:{upper_bound}):")
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.start.setRange(min_, max_)
        self.end.setRange(min_, max_)

    def setValue(self, value: tuple[int, int]):
        self.start.setValue(max(value[0], self.lower_bound))
        self.end.setValue(min(value[1], self.upper_bound))

    def value(self) -> tuple[int, int]:
        return (
            max(self.start.value(), self.lower_bound),
            min(self.end.value(), self.upper_bound),
        )


class ApparentGroup(QGroupBox):
    def __init__(self, parent):
        super().__init__(title="Apparent", parent=parent)
        self.setMinimumHeight(60)
        self.setMinimumWidth(108)

        self.label_color = ColorButton(self, color="yellow")
        self.label_color.setText("Label")

        lb1 = QLabel(text="Font Size (pt)", parent=self)
        self.label_size = QSpinBox(self)
        self.label_size.setSingleStep(1)
        self.label_size.setMinimum(8)
        self.label_size.setMaximum(512)
        self.label_size.setValue(16)

        self.nose_color = ColorButton(self, color="red")
        self.nose_color.setText("Nose")

        self.body_color = ColorButton(self, color="yellow")
        self.body_color.setText("Body")
        lb2 = QLabel(text="Body Thickness", parent=self)
        self.line_width = QSpinBox(self)
        self.line_width.setSingleStep(1)
        self.line_width.setMinimum(1)
        self.line_width.setMaximum(50)
        self.line_width.setValue(3)

        vlayout = QVBoxLayout()

        vlayout.addWidget(self.label_color)
        vlayout.addWidget(lb1)
        vlayout.addWidget(self.label_size)
        vlayout.addWidget(self.nose_color)
        vlayout.addWidget(self.body_color)
        vlayout.addWidget(lb2)
        vlayout.addWidget(self.line_width)

        self.setLayout(vlayout)

    def get_status(self) -> dict[str, Any]:
        return {
            "label_size": self.label_size.value(),
            "line_width": self.line_width.value(),
            "label_color": self.label_color.color(),
            "nose_color": self.nose_color.color(),
            "body_color": self.body_color.color(),
        }

    def set_status(self, status: dict[str, Any]):
        self.label_size.setValue(status["label_size"])
        self.line_width.setValue(status["line_width"])
        self.label_color.setColor(status["label_color"])
        self.nose_color.setColor(status["nose_color"])
        self.body_color.setColor(status["body_color"])


class WormTracerUI(QWidget):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        parent=None,
    ):  # type-hint is required
        super().__init__(parent)
        self.session_code = get_barcode()

        self._viewer: napari.Viewer = viewer
        # QPushButton (name, callback)
        btns = [
            ("Load Image", self._load_image),
            ("Load Centerline", self._load_centerline),
            ("Resume Status", self._resume_status),
            ("Save Status", self._save_status),
            # ("Prev (-1)", functools.partial(self._move_frame, step=-1)),
            # ("Next (+1)", functools.partial(self._move_frame, step=1)),
            ("Refetch Centerline", self._ask_for_refetch),
            ("Flip Head/Tail", self._flip),
            ("Reset Centerline", self._reset_centerline),
            ("Register", self._register),
            ("Undo Modify", self._undo),
            ("Export Centerline", self._save_to_file),
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

        self.range = GuideRangeGroup("Mark as Guide", parent=self)
        self.range.label_btn.pressed.connect(self._label_as_guide)

        n_btns = len(btns)
        row_idx = n_btns // ncol
        col_idx = n_btns % ncol

        layout.addWidget(
            self.range, row_idx, col_idx, Qt.AlignmentFlag.AlignCenter
        )
        n_btns += 1

        group_box = QGroupBox("Export As", self)
        group_box.setMinimumHeight(80)
        group_box.setMinimumWidth(108)

        vlayout = QVBoxLayout()
        group_box.setLayout(vlayout)

        self.group_buttons = QButtonGroup(self)
        self.group_buttons.setExclusive(True)

        check_btn = QRadioButton(self)
        check_btn.setText("hdf")
        check_btn.setChecked(True)
        self.group_buttons.addButton(check_btn)
        vlayout.addWidget(check_btn)

        check_btn = QRadioButton(self)
        check_btn.setText("csv")
        self.group_buttons.addButton(check_btn)
        vlayout.addWidget(check_btn)

        lb = QLabel(parent=self, text="plot_n:")
        vlayout.addWidget(lb)
        self.plot_n = QSpinBox(self)
        self.plot_n.setMinimum(10)
        self.plot_n.setMaximum(9999)
        vlayout.addWidget(self.plot_n)

        self.save_guide = QCheckBox(self)
        self.save_guide.setText("With Guide")
        vlayout.addWidget(self.save_guide)

        row_idx = n_btns // ncol
        col_idx = n_btns % ncol

        layout.addWidget(
            group_box, row_idx, col_idx, Qt.AlignmentFlag.AlignCenter
        )
        n_btns += 1

        self.apparent = ApparentGroup(parent=self)

        self.apparent.label_color.colorChanged.connect(
            lambda color: self._update_color(color, target_shape="label")
        )
        self.apparent.nose_color.colorChanged.connect(
            lambda color: self._update_color(color, target_shape="nose")
        )
        self.apparent.body_color.colorChanged.connect(
            lambda color: self._update_color(color, target_shape="body")
        )
        self.apparent.line_width.valueChanged.connect(self._update_width)
        self.apparent.label_size.valueChanged.connect(
            self._update_lbl_font_size
        )

        row_idx = n_btns // ncol
        col_idx = n_btns % ncol

        layout.addWidget(
            self.apparent, row_idx, col_idx, Qt.AlignmentFlag.AlignCenter
        )
        n_btns += 1

        layout.setSpacing(0)
        layout.setContentsMargins(16, 8, 16, 8)
        self.setLayout(layout)

        self.centerlines = None
        self.state = None
        self.is_flip = None
        # Memory the unmodified line. for redo
        self.history = []
        # src_path
        self.src_path = None

        # layers
        self.body_layer = None
        self.nose_layer = None

    def _move_frame(self, step: int):
        z_idx = self._viewer.dims.current_step[0]
        # reset current index
        self._reset_centerline()
        next_step = max(z_idx + step, 0)
        if self.centerlines is not None:
            T = self.centerlines.shape[0]
            next_step = min(next_step, T - 1)
        self._viewer.dims.set_current_step(0, next_step)

    def _resume_status(self):
        status_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select .pickle to resume the previous status",
            filter="Pickle Files (*.pkl *.pickle );;All Files (*.*)",
        )

        path = Path(status_path)
        try:
            with path.open("rb") as fd:
                status = pickle.load(fd)
        except Exception as _:  # noqa: BLE001
            show_error(f"Fail to load status file: {path}")
            return

        # Retrieve session_code if possible
        m = re.search(r"\.(\w{8})\.(pickle)|(pkl)$", path.name)
        if m is not None:
            self.session_code = m.group(1)

        self.apparent.set_status(status["apparent"])
        x_src, y_src = tuple(map(Path, status["src_path"]))
        if not x_src.is_file():
            # If x_src is not existed.
            # We will try to load the data from the pickle directory.
            x_src = path.parent / x_src.name
            y_src = path.parent / y_src.name

        if not (x_src.is_file() and y_src.is_file()):
            show_info(f"Cannot file centerline data from: {path}")
            return

        self.src_path = tuple(map(Path, status["src_path"]))

        self._refetch_centerline()
        # Restore the status and history
        self.state = status["state"]
        self.history = status["history"]
        self.is_flip = status["is_flip"]
        self._viewer.dims.set_current_step(0, status["z_idx"])
        self._viewer.camera.zoom = status["zoom"]
        self._viewer.camera.center = status["center"]

        if (
            self.body_layer is not None
            and self.nose_layer is not None
            and "face_color" in status
        ):
            self.body_layer.face_color = status["face_color"]
            self.body_layer.edge_color = status["body_ec"]
            self.body_layer.edge_width = status["body_width"]

            self.nose_layer.face_color = status["face_color"]
            self.nose_layer.border_color = status["nose_color"]
            self.nose_layer.size = status["nose_sz"]
            self.nose_layer.border_width = status["nose_sz_width"]

        self._reset_centerline()

    def _save_status(self):
        suffix = self.history[-1][0] if self.history else self.session_code
        init_filepath = Path.cwd() / f"status.{suffix}.pickle"
        x_src = ""
        y_src = ""
        if self.src_path is not None:
            x_src, y_src = self.src_path
            parent = x_src.parent
            prefix = os.path.commonprefix([x_src.stem, y_src.stem]).strip("_")
            pat = r"_(x|y|xy)"
            # Remove all _x or _y
            prefix = re.sub(pat, "", prefix)
            # Remove the timestamp barcode if existes.
            prefix = prefix.split(".")[0]
            init_filepath = parent / f"{prefix}_status.{suffix}.pickle"

            x_src = os.fspath(x_src)
            y_src = os.fspath(y_src)

        outputfile, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save current status as pickle...",
            directory=os.fspath(init_filepath),
            filter=" Pickle Files (*.pkl *.pickle);;All Files (*.*)",
        )

        if not os.path.isfile(outputfile):
            return

        status = {
            "src_path": (x_src, y_src),
            "state": self.state,
            "history": self.history,
            "is_flip": self.is_flip,
            # Napari status
            "z_idx": self._viewer.dims.current_step[0],
            "zoom": self._viewer.camera.zoom,
            "center": self._viewer.camera.center,
            # Apparent
            "apparent": self.apparent.get_status(),
        }

        if self.body_layer is not None and self.nose_layer is not None:
            # Layer status
            status.update(
                face_color=self.body_layer.face_color,
                body_ec=self.body_layer.edge_color,
                body_width=self.body_layer.edge_width,
                nose_color=self.nose_layer.border_color,
                nose_sz=self.nose_layer.size,
                nose_sz_width=self.nose_layer.border_width,
            )

        with open(outputfile, mode="wb") as fd:
            pickle.dump(status, fd)

    def _save_to_file(self):
        if self.centerlines is None or self.src_path is None:
            return
        assert self.is_flip is not None, ""

        if not self.history:
            reply = QMessageBox.warning(
                self,
                "Warning: No modifications found",
                "You haven't made any changes yet.\nDo you still want to save?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # Default to No for safety
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Z, Y, X
        x = self.centerlines[:, :, 2].copy()
        y = self.centerlines[:, :, 1].copy()

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
        # We update the barcode only if the editing happens
        suffix = self.history[-1][0] if self.history else self.session_code
        pat = r"_(x|y|xy)"
        # Remove all _x or _y
        prefix = re.sub(pat, "", prefix)
        # Remove the timestamp barcode if existes.
        prefix = prefix.split(".")[0]

        plot_n = x.shape[1]
        if self.plot_n.value() != plot_n:
            cs = interpolate.CubicSpline(np.linspace(0.0, 1.0, plot_n), x)
            x = cs(np.linspace(0.0, 1.0, self.plot_n.value()))
            cs = interpolate.CubicSpline(np.linspace(0.0, 1.0, plot_n), y)
            y = cs(np.linspace(0.0, 1.0, self.plot_n.value()))

        if output_type == "csv":
            # Remove all x and y
            x_dst = parent.joinpath(f"{prefix}_x.{suffix}.csv")
            y_dst = parent.joinpath(f"{prefix}_y.{suffix}.csv")
            np.savetxt(x_dst, x, delimiter=",")
            np.savetxt(y_dst, y, delimiter=",")
        else:
            dst = parent.joinpath(f"{prefix}.{suffix}.h5")
            with h5py.File(dst, "w") as handler:
                handler.create_dataset("x", data=x)
                handler.create_dataset("y", data=y)

        show_info("Modified centerline was saved.")

        if self.save_guide.isChecked():
            assert self.state is not None, ""
            guide_frame = self.state > 0
            x[~guide_frame] = np.nan
            y[~guide_frame] = np.nan

            prefix = prefix + "_guide"

            if output_type == "csv":
                # Remove all x and y
                x_dst = parent.joinpath(f"{prefix}_x.{suffix}.csv")
                y_dst = parent.joinpath(f"{prefix}_y.{suffix}.csv")
                np.savetxt(x_dst, x, delimiter=",")
                np.savetxt(y_dst, y, delimiter=",")
            else:
                dst = parent.joinpath(f"{prefix}.{suffix}.h5")
                with h5py.File(dst, "w") as handler:
                    handler.create_dataset("x", data=x)
                    handler.create_dataset("y", data=y)

            show_info("Guided centerline was saved.")

    def _label_as_guide(self):
        if self.centerlines is None:
            return
        assert self.state is not None, ""
        assert self.body_layer is not None, ""
        n_frame = self.centerlines.shape[0]
        start, end = self.range.value()
        start, end = np.clip((start, end), 0, n_frame).astype(int)
        # Inclusive
        self.state[start : end + 1] += 1
        z_idx = self._viewer.dims.current_step[0]
        self.history.append((get_barcode(), z_idx, (start, end)))

        features = self.body_layer.features
        features["state"] = self.state
        features["state_label"] = np.array(["Raw", "Guide"])[
            (self.state > 0).astype(int)
        ]
        self.body_layer.features = features
        self.body_layer.refresh_text()

    def _ask_for_refetch(self):
        if self.src_path is None:
            return

        reply = QMessageBox.warning(
            self,
            "Warning: Data Loss",
            "Refetching will overwrite all manual changes. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._refetch_centerline()

    def _load_centerline(self):
        x_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select .csv file generated by WormTracer (ex: *_x.csv)",
            filter="CSV or HDF Files (*.csv *.h5 *.hdf);;All Files (*.*)",
        )
        if not x_path:
            return

        folder = Path(x_path).parent
        name = Path(x_path).name
        if name.endswith(".h5"):
            x_name = name
            y_name = name
        elif "_x" in name:
            x_name = name
            y_name = find_most_commonprefix_name(folder, name)
        elif "_y" in name:
            y_name = name
            x_name = find_most_commonprefix_name(folder, name)
        else:
            if self.centerlines is None:
                show_error(
                    f"Select file did not contains proper suffixed by `_x` or `_y`: {name}"
                )
            return

        self.src_path = (folder.joinpath(x_name), folder.joinpath(y_name))
        self._refetch_centerline()

    def _refetch_centerline(self):
        if self._viewer is None or self.src_path is None:
            return

        z_idx = 0
        face_color = "transparent"
        body_ec = "yellow"
        body_width = 2
        nose_color = "red"
        nose_sz = 5
        nose_sz_width = 0.15
        text = {
            "string": "{index}: {state_label} (+{state})",
            "anchor": "upper_left",
            "translation": [0, -8, 0],  # Z, Y, X
            "size": 16,  # fontsize
            "color": "yellow",
        }
        zoom = self._viewer.camera.zoom
        center = self._viewer.camera.center

        if self.body_layer is not None:
            face_color = self.body_layer.face_color
            body_ec = self.body_layer.edge_color
            body_width = self.body_layer.edge_width
            assert self.nose_layer is not None
            nose_color = self.nose_layer.border_color
            nose_sz = self.nose_layer.size
            nose_sz_width = self.nose_layer.border_width

            z_idx = self._viewer.dims.current_step[0]
            text = dict(self.body_layer.text)
            self._viewer.layers.remove(self.body_layer)
            self._viewer.layers.remove(self.nose_layer)
            assert self.nose_layer is not None, ""
            self.body_layer = None
            self.nose_layer = None

        x_src, y_src = self.src_path
        if x_src.name.endswith(".csv"):
            # load x and y
            x = np.loadtxt(x_src, delimiter=",")
            y = np.loadtxt(y_src, delimiter=",")
        elif x_src.name.endswith(".h5"):
            with h5py.File(x_src, "r") as handler:
                x = np.asarray(handler["x"])
                y = np.asarray(handler["y"])
        else:
            raise ValueError("src_path must be .csv or .h5 files")

        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)
        x = np.where(np.isfinite(x), x, x_mean)
        y = np.where(np.isfinite(y), y, y_mean)

        T, plot_n = x.shape
        self.range.setRange(0, T * 10, 0, T - 1)
        self.range.setValue((T // 5, T // 5 * 4))

        self.plot_n.setValue(plot_n)

        z = np.repeat(np.arange(T), plot_n).reshape(T, plot_n)
        self.centerlines = np.stack([z, y, x], axis=-1)  # (1500, 100, 3)
        # Clean state and history
        self.state = np.zeros(T, dtype=int)
        self.history = []

        self.body_layer = self._viewer.add_shapes(
            data=list(self.centerlines),
            ndim=3,
            shape_type="path",  # 'path' means polyline in napari
            name="centerline",
            # This parameter is crucial: it tells napari how to group the vertices
            # into separate shapes (one shape per time point in this case)
            face_color=face_color,
            edge_color=body_ec,
            edge_width=body_width,
            features={
                "index": np.arange(len(self.centerlines)),
                "state_label": np.array(["Raw", "Guide"])[
                    (self.state > 0).astype(int)
                ],
                "state": self.state,
            },
            text=text,
        )

        self.body_layer.editable = True
        # (T, 1, 3) => (T, 4, 3)
        # [4, 3] => [1, 4, 3]

        self.nose_layer = self._viewer.add_points(
            data=[skel[0] for skel in self.centerlines],
            ndim=3,
            name="nose",
            face_color=face_color,
            border_color=nose_color,
            size=nose_sz,
            border_width=nose_sz_width,
        )
        self.nose_layer.editable = False
        # memory whether current centerline was flip or not
        self.is_flip = np.repeat(0, T).astype("u1")
        self._viewer.dims.set_current_step(0, z_idx)
        self._viewer.camera.zoom = zoom
        if center is not None:
            self._viewer.camera.center = center
        self.body_layer.refresh_text()

    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select an Image File",
            filter="Image Files (*.png *.tif *.tiff *.jpg *.avi *.mp4);;All Files (*.*)",
        )
        if not file_path:
            return

        im_layer = self._viewer.open(file_path, stack=True)
        current_index = self._viewer.layers.index(im_layer[0])
        # Move the image to the button
        self._viewer.layers.move(current_index, 0)

    def _register(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, "Some problem occurs"
        assert self.body_layer is not None, ""
        assert self.nose_layer is not None, ""
        assert self.state is not None, ""

        z_idx = self._viewer.dims.current_step[0]

        data = self.body_layer.data
        # Get all shapes associated to current indices
        current_data = [
            (i, d)
            for (i, d) in enumerate(data)
            if d[0, 0] == z_idx and d.shape[0] > 1
        ]

        if not current_data:
            return

        if len(current_data) > 1:
            # If previous shape exists, drop the previous shape
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
        self.history.append(
            (get_barcode(), z_idx, self.centerlines[z_idx].copy())
        )
        # Assign the interpolated data to centerline
        self.centerlines[z_idx] = interpolated_data
        # reset flip
        self.is_flip[z_idx] = 0
        self.state[z_idx] += 1
        # Update the centerline using reset_centerline
        self._reset_centerline()

    def _flip(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, ""
        assert self.state is not None, ""

        z_idx = self._viewer.dims.current_step[0]
        # Make sure the append type is (int, int)
        self.history.append((get_barcode(), z_idx, int(self.is_flip[z_idx])))
        self.is_flip[z_idx] = self.is_flip[z_idx] ^ 1
        self.state[z_idx] += 1
        self._reset_centerline()

    def _reset_centerline(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, ""
        assert self.body_layer is not None, ""
        assert self.nose_layer is not None, ""
        assert self.state is not None, ""

        centerlines = self.centerlines.copy()
        mask = self.is_flip != 0
        centerlines[mask] = centerlines[mask, ::-1, :]

        # we have to assign the data to update the drawing
        self.body_layer.data = centerlines
        features = self.body_layer.features
        features["state"] = self.state
        features["state_label"] = np.array(["Raw", "Guide"])[
            (self.state > 0).astype(int)
        ]
        self.body_layer.features = features
        self.body_layer.refresh_text()
        # we have to assign the data to update the drawing
        self.nose_layer.data = centerlines[:, 0, :]

    def _undo(self):
        if self.centerlines is None:
            return
        assert self.is_flip is not None, ""
        assert self.state is not None, ""
        try:
            _, z_idx, prev_skel = self.history.pop()
            z_idx = int(z_idx)
            self._viewer.dims.set_current_step(0, z_idx)
            if isinstance(prev_skel, int):
                # If previous step is flip, we just revert the flip.
                self.is_flip[z_idx] = prev_skel
                self.state[z_idx] -= 1
            elif isinstance(prev_skel, tuple):
                start, end = prev_skel
                self.state[start : end + 1] -= 1
            elif isinstance(prev_skel, np.ndarray):
                self.centerlines[z_idx] = prev_skel
                self.state[z_idx] -= 1

        except IndexError:
            # Pop item from empty history list will raise IndexError.
            self.state[:] = 0
        self._reset_centerline()

    def _update_color(self, color, target_shape=""):
        if self.body_layer is None or self.nose_layer is None:
            return

        if target_shape == "nose":
            self.nose_layer.border_color = color
            self.nose_layer.refresh_colors()
            self.nose_layer.face_color = "transparent"
            self.nose_layer.refresh_colors()

        if target_shape == "body":
            self.body_layer.edge_color = color
            self.body_layer.refresh_colors()

        if target_shape == "label":
            prop = dict(self.body_layer.text)
            prop["color"] = color
            self.body_layer.text = prop
            self.body_layer.refresh_text()

    def _update_width(self, value):
        if self.body_layer is not None:
            self.body_layer.edge_width = value
            self.body_layer.refresh()

    def _update_lbl_font_size(self, value):
        if self.body_layer is not None:
            text = dict(self.body_layer.text)
            text["size"] = value
            self.body_layer.text = text
            self.body_layer.refresh_text()
