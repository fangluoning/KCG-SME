import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.config import get_config  # noqa: E402
from models.kcg_sme_model import KCGSMEModel  # noqa: E402


EMG_SECTIONS = {
    "Leg EMG": slice(6, 10),
    "Upper Arm EMG": slice(19, 27),
    "Lower Arm EMG": slice(27, 35),
}

JOINT_SECTIONS = {
    "Left Foot Pressure": slice(0, 3),
    "Right Foot Pressure": slice(3, 6),
    "Hip Rotation": slice(10, 13),
    "Spine Rotation": slice(13, 16),
    "Shoulder Rotation": slice(16, 19),
    "Hand Euler": slice(35, 38),
}

SKILL_LABELS = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}

LANG_TEXT = {
    "en": {
        "sample_label": "Sample Index:",
        "language": "Language:",
        "analyze": "Analyze Sample",
        "status_ready": "Ready",
        "status_loading": "Loading sample...",
        "status_plotting": "Rendering plots...",
        "status_done": "Analysis complete",
        "summary_title": "[Evaluation Summary]",
        "probabilities": "Probabilities",
        "true_label": "True label",
        "pred_label": "Predicted label",
        "progress_title": "Processing",
    },
}

SEGMENT_LABELS = [
    "Hips",
    "Right UpLeg",
    "Right Leg",
    "Right Foot",
    "Left UpLeg",
    "Left Leg",
    "Left Foot",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "Right Shoulder",
    "Right Arm",
    "Right ForeArm",
    "Right Hand",
    "Left Shoulder",
    "Left Arm",
    "Left ForeArm",
    "Left Hand",
]

SEGMENT_CHAINS = {
    "Left Legs": ["Hips", "Left UpLeg", "Left Leg", "Left Foot"],
    "Right Legs": ["Hips", "Right UpLeg", "Right Leg", "Right Foot"],
    "Spine": ["Head", "Neck1", "Neck", "Spine2", "Spine1", "Spine", "Hips"],
    "Left Arm": ["Left Shoulder", "Left Arm", "Left ForeArm", "Left Hand"],
    "Right Arm": ["Right Shoulder", "Right Arm", "Right ForeArm", "Right Hand"],
}


class SampleAnalyzer:
    def __init__(self):
        self.cfg = get_config()
        self.device = torch.device(self.cfg.device)
        self.model = KCGSMEModel(self.cfg.model_config).to(self.device)
        self.model.load_state_dict(torch.load(self.cfg.checkpoint_path, map_location=self.device))
        self.model.eval()
        self.data_path = self.cfg.data_path
        with h5py.File(self.data_path, "r") as f:
            self.total_samples = f["feature_matrices"].shape[0]
            self.skill_levels = f["skill_levels"][:]
            if "example_subject_ids" in f:
                self.subject_ids = [sid.decode("utf-8") for sid in f["example_subject_ids"][:]]
            else:
                self.subject_ids = [f"sample_{idx:04d}" for idx in range(self.total_samples)]
        skeleton_path = Path("data_processed/data_processed_allStreams_60hz_onlyForehand_skeleton_skill_level.hdf5")
        self.skeleton_data = None
        if skeleton_path.exists():
            self.skeleton_file = h5py.File(skeleton_path, "r")
        else:
            self.skeleton_file = None

    def load_sample(self, index: int):
        with h5py.File(self.data_path, "r") as f:
            features = f["feature_matrices"][index]
            label = int(self.skill_levels[index])
        skeleton = None
        if self.skeleton_file is not None:
            skeleton = self.skeleton_file["feature_matrices"][index]
        sequence = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _, _ = self.model(sequence)
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        preds = logits.argmax(dim=-1)
        record = {
            "true_label": label,
            "pred_label": preds[0].item(),
            "probabilities": probs.tolist(),
        }
        return {
            "features": features,
            "skeleton": skeleton,
            "record": record,
            "subject": self.subject_ids[index] if index < len(self.subject_ids) else "Unknown",
            "label_text": SKILL_LABELS.get(label, str(label)),
            "pred_text": SKILL_LABELS.get(preds[0].item(), str(preds[0].item())),
        }


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setStyleSheet("background-color: white;")


class MatplotlibCanvas3D(FigureCanvas):
    def __init__(self, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setStyleSheet("background-color: white;")


class SkillVisualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.analyzer = SampleAnalyzer()
        self.setWindowTitle("KCG-SME Skill Visualizer")
        self.resize(1400, 900)
        self.chain_indexes = self._build_chain_indexes()
        self.skel_timer = QtCore.QTimer()
        self.skel_timer.timeout.connect(self._advance_skeleton_frame)
        self.language = "en"
        self.play_btn = None
        self._setup_ui()
        self.current_index = 0

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        control_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(control_layout)

        self.index_spin = QtWidgets.QSpinBox()
        self.index_spin.setRange(0, self.analyzer.total_samples - 1)
        self.index_spin.setValue(0)
        self.sample_label = QtWidgets.QLabel()
        control_layout.addWidget(self.sample_label)
        control_layout.addWidget(self.index_spin)

        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(["English"])
        self.lang_combo.currentIndexChanged.connect(self.on_language_change)
        control_layout.addWidget(self.lang_combo)

        self.analyze_btn = QtWidgets.QPushButton()
        self.analyze_btn.clicked.connect(self.on_analyze)
        control_layout.addWidget(self.analyze_btn)

        self.status_label = QtWidgets.QLabel()
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        # three-column layout
        plots_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(plots_layout, stretch=3)

        self.emg_canvas = MatplotlibCanvas(width=6, height=4)
        plots_layout.addWidget(self.emg_canvas, stretch=1)

        self.joint_canvas = MatplotlibCanvas(width=6, height=4)
        plots_layout.addWidget(self.joint_canvas, stretch=1)

        right_panel = QtWidgets.QVBoxLayout()
        plots_layout.addLayout(right_panel, stretch=1)

        self.skeleton_canvas = MatplotlibCanvas3D(width=4, height=3)
        right_panel.addWidget(self.skeleton_canvas, stretch=1)
        play_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton()
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_skeleton_play)
        play_layout.addWidget(self.play_btn)
        play_layout.addStretch()
        right_panel.addLayout(play_layout)
        self.summary_box = QtWidgets.QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(260)
        right_panel.addWidget(self.summary_box, stretch=1)
        self.progress = None
        self._apply_language()

    def on_language_change(self, idx):
        self.language = "en"
        self._apply_language()

    def _apply_language(self):
        texts = LANG_TEXT[self.language]
        self.sample_label.setText(texts["sample_label"])
        self.analyze_btn.setText(texts["analyze"])
        self.status_label.setText(texts["status_ready"])
        if self.play_btn is not None:
            if self.skel_timer.isActive():
                self.play_btn.setText("Pause")
            else:
                self.play_btn.setText("Play Skeleton")

    def on_analyze(self):
        idx = self.index_spin.value()
        self._set_status("status_loading")
        self._show_progress("status_loading")
        QtWidgets.QApplication.processEvents()
        result = self.analyzer.load_sample(idx)
        self.current_data = result
        self._set_status("status_plotting")
        self._update_progress("status_plotting")
        self._plot_emg(result["features"])
        self._plot_joint(result["features"])
        self._plot_skeleton(result.get("skeleton"))
        self._close_progress()
        summary = [
            f"{LANG_TEXT[self.language]['summary_title']}",
            f"Index: {idx} | Subject: {result['subject']}",
            f"{LANG_TEXT[self.language]['true_label']}: {result['label_text']} | "
            f"{LANG_TEXT[self.language]['pred_label']}: {result['pred_text']}",
            f"{LANG_TEXT[self.language]['probabilities']}: {result['record']['probabilities']}",
        ]
        self.summary_box.setPlainText("\n".join(summary))
        self._set_status("status_done")

    def _plot_emg(self, features):
        self.emg_canvas.fig.clf()
        time_axis = np.linspace(0, 2.5, features.shape[0])
        for idx, (title, slc) in enumerate(EMG_SECTIONS.items(), start=1):
            ax = self.emg_canvas.fig.add_subplot(len(EMG_SECTIONS), 1, idx)
            section = features[:, slc]
            for channel in range(section.shape[1]):
                ax.plot(time_axis, section[:, channel], linewidth=1, alpha=0.8)
            ax.set_ylabel(title, fontname="Times New Roman")
            ax.grid(color="#b0b0b0", linestyle="-", linewidth=0.5)
        self.emg_canvas.fig.tight_layout()
        self.emg_canvas.draw()

    def _plot_joint(self, features):
        self.joint_canvas.fig.clf()
        time_axis = np.linspace(0, 2.5, features.shape[0])
        for idx, (title, slc) in enumerate(JOINT_SECTIONS.items(), start=1):
            ax = self.joint_canvas.fig.add_subplot(len(JOINT_SECTIONS), 1, idx)
            section = features[:, slc]
            for channel in range(section.shape[1]):
                ax.plot(time_axis, section[:, channel], linewidth=1.2)
            ax.set_ylabel(title, fontname="Times New Roman")
            ax.grid(color="#b0b0b0", linestyle="-", linewidth=0.5)
        self.joint_canvas.fig.tight_layout()
        self.joint_canvas.draw()

    def _plot_skeleton(self, skeleton):
        if self.skel_timer.isActive():
            self.skel_timer.stop()
        self.skeleton_data = skeleton
        ax = self.skeleton_canvas.ax
        ax.clear()
        if skeleton is None:
            ax.text2D(0.2, 0.5, "No skeleton data", transform=ax.transAxes)
            self.skeleton_canvas.draw()
            if self.play_btn is not None:
                self.play_btn.setEnabled(False)
                self.play_btn.setText("Play Skeleton")
            return
        if self.play_btn is not None and not self.play_btn.isEnabled():
            self.play_btn.setEnabled(True)
        frames = skeleton.shape[0]
        self.skeleton_frame = 0
        points = skeleton[self.skeleton_frame].reshape(-1, 3)
        for indexes in self.chain_indexes.values():
            if not indexes:
                continue
            seg = points[indexes, :]
            ax.plot(seg[:, 2], seg[:, 0], seg[:, 1], "-o", markersize=4)
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        ax.set_xlim(np.min(points[:, 2]) - 50, np.max(points[:, 2]) + 50)
        ax.set_ylim(np.min(points[:, 0]) - 50, np.max(points[:, 0]) + 50)
        ax.set_zlim(0, np.max(points[:, 1]) + 100)
        ax.view_init(elev=20, azim=-70)
        self.skeleton_canvas.draw()
        interval = max(15, int(2500 / max(frames, 1)))
        self.skel_timer.start(interval)
        if self.play_btn is not None:
            self.play_btn.setText("Pause")

    def toggle_skeleton_play(self):
        if self.skeleton_data is None or self.play_btn is None or not self.play_btn.isEnabled():
            return
        if self.skel_timer.isActive():
            self.skel_timer.stop()
            self.play_btn.setText("Play Skeleton")
        else:
            frames = self.skeleton_data.shape[0]
            interval = self.skel_timer.interval() or max(15, int(2500 / max(frames, 1)))
            self.skel_timer.start(interval)
            self.play_btn.setText("Pause")

    def _advance_skeleton_frame(self):
        if self.skeleton_data is None:
            self.skel_timer.stop()
            if self.play_btn is not None:
                self.play_btn.setEnabled(False)
                self.play_btn.setText("Play Skeleton")
            return
        self.skeleton_frame = (self.skeleton_frame + 1) % self.skeleton_data.shape[0]
        ax = self.skeleton_canvas.ax
        ax.clear()
        points = self.skeleton_data[self.skeleton_frame].reshape(-1, 3)
        for indexes in self.chain_indexes.values():
            if not indexes:
                continue
            seg = points[indexes, :]
            ax.plot(seg[:, 2], seg[:, 0], seg[:, 1], "-o", markersize=4)
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        ax.set_xlim(np.min(points[:, 2]) - 50, np.max(points[:, 2]) + 50)
        ax.set_ylim(np.min(points[:, 0]) - 50, np.max(points[:, 0]) + 50)
        ax.set_zlim(0, np.max(points[:, 1]) + 100)
        ax.view_init(elev=20, azim=-70)
        self.skeleton_canvas.draw()

    def _set_status(self, key: str):
        self.status_label.setText(LANG_TEXT[self.language].get(key, key))

    def _build_chain_indexes(self):
        mapping = {}
        for chain, labels in SEGMENT_CHAINS.items():
            idxs = []
            for label in labels:
                if label in SEGMENT_LABELS:
                    idxs.append(SEGMENT_LABELS.index(label))
            mapping[chain] = idxs
        return mapping

    def _show_progress(self, key: str):
        if self.progress:
            self.progress.close()
        self.progress = QtWidgets.QProgressDialog(
            LANG_TEXT[self.language]["progress_title"],
            None,
            0,
            0,
            self,
        )
        self.progress.setWindowTitle(LANG_TEXT[self.language]["progress_title"])
        self.progress.setLabelText(LANG_TEXT[self.language].get(key, ""))
        self.progress.setCancelButton(None)
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.show()

    def _update_progress(self, key: str):
        if self.progress:
            self.progress.setLabelText(LANG_TEXT[self.language].get(key, ""))
            QtWidgets.QApplication.processEvents()

    def _close_progress(self):
        if self.progress:
            self.progress.close()
            self.progress = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    viewer = SkillVisualizer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
