"""
Roulette Speed Detector v6.1
自动测速：预热3个亮片后开始计数
跨平台版本 (macOS/Windows)
"""

import sys
import json
import time
import os
import platform
from collections import deque
import numpy as np
import cv2
from mss import mss
from pynput import keyboard
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QCursor, QPainter, QColor, QPen

# ==================== 平台检测 ====================
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

# ==================== Windows DPI 修复 ====================
if IS_WINDOWS:
    try:
        import ctypes

        # 让程序感知 DPI，获取真实的屏幕坐标
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# ==================== 路径处理（支持打包后运行）====================
def get_base_path():
    """获取程序运行的基础路径，支持 PyInstaller 打包"""
    if getattr(sys, 'frozen', False):
        # 打包后运行
        return os.path.dirname(sys.executable)
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))


def get_config_path():
    """获取配置文件的完整路径"""
    return os.path.join(get_base_path(), "roulette_config.json")


# ==================== Config ====================
DEFAULT_CONFIG = {
    "detect_point": {"x": 400, "y": 300},
    "sample_size": 5,
    "threshold_pct": 20,
    "moving_avg_window": 30,
    "warmup_count": 5,
    "slots_to_count": 6,
    "min_peak_interval": 0.04,
}


def load_config():
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = DEFAULT_CONFIG.copy()
                cfg.update(json.load(f))
                return cfg
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config):
    config_path = get_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save config: {e}")


# ==================== Screen Capture ====================
class Screen:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
            cls._inst.sct = mss()
            cls._inst._init_dpi_scale()
        return cls._inst

    def _init_dpi_scale(self):
        """初始化 DPI 缩放因子"""
        self.dpi_scale = 1.0
        if IS_WINDOWS:
            try:
                # 在 Windows 上，如果 DPI 感知设置正确，这里应该是 1.0
                # 但为了安全起见，我们检测一下
                import ctypes
                user32 = ctypes.windll.user32
                # 获取主显示器的 DPI
                hdc = user32.GetDC(0)
                gdi32 = ctypes.windll.gdi32
                dpi = gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
                user32.ReleaseDC(0, hdc)
                self.dpi_scale = dpi / 96.0
                print(f"[INFO] Windows DPI: {dpi}, Scale: {self.dpi_scale}")
            except Exception as e:
                print(f"[WARN] Could not detect DPI: {e}")

    def get_brightness(self, x, y, size=5):
        try:
            half = size // 2
            # 注意：mss 使用的是物理像素坐标
            # 由于我们已经设置了 DPI Awareness，x/y 应该是正确的物理坐标
            monitor = {
                "left": x - half,
                "top": y - half,
                "width": size,
                "height": size
            }
            img = np.array(self.sct.grab(monitor))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return float(np.mean(gray))
        except Exception as e:
            print(f"[ERROR] Screen capture failed: {e}")
            return 0


# ==================== Brightness Graph ====================
class BrightnessGraph(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)
        self.setStyleSheet("background: #1a1a2e; border-radius: 6px;")
        self.data = deque(maxlen=200)
        self.avg_line = deque(maxlen=200)

    def add_point(self, brightness, moving_avg, is_peak=False):
        self.data.append((brightness, is_peak))
        self.avg_line.append(moving_avg)
        self.update()

    def clear(self):
        self.data.clear()
        self.avg_line.clear()
        self.update()

    def paintEvent(self, event):
        if len(self.data) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin = 8

        all_values = [d[0] for d in self.data] + list(self.avg_line)
        if not all_values:
            return
        min_v = min(all_values) - 5
        max_v = max(all_values) + 5
        range_v = max(max_v - min_v, 1)

        def to_y(v):
            return h - margin - (v - min_v) / range_v * (h - 2 * margin)

        def to_x(i):
            return margin + i * (w - 2 * margin) / max(len(self.data), 1)

        if len(self.avg_line) > 1:
            painter.setPen(QPen(QColor(255, 100, 100), 1, Qt.DashLine))
            avg_points = [(to_x(i), to_y(v)) for i, v in enumerate(self.avg_line) if v > 0]
            for i in range(1, len(avg_points)):
                painter.drawLine(int(avg_points[i - 1][0]), int(avg_points[i - 1][1]),
                                 int(avg_points[i][0]), int(avg_points[i][1]))

        painter.setPen(QPen(QColor(100, 255, 150), 2))
        data_list = list(self.data)
        points = [(to_x(i), to_y(d[0])) for i, d in enumerate(data_list)]

        for i in range(1, len(points)):
            painter.drawLine(int(points[i - 1][0]), int(points[i - 1][1]),
                             int(points[i][0]), int(points[i][1]))

        painter.setPen(QPen(QColor(255, 255, 0), 2))
        painter.setBrush(QColor(255, 255, 0))
        for i, (b, is_peak) in enumerate(data_list):
            if is_peak:
                px, py = points[i]
                painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)

        painter.setPen(QColor(150, 150, 150))
        painter.drawText(10, 15, f"Range: {min_v:.0f} - {max_v:.0f}")


# ==================== Detector Thread ====================
class DetectorThread(QThread):
    status = pyqtSignal(str)
    result = pyqtSignal(float, float, int)
    brightness = pyqtSignal(float, float, bool)
    debug = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.screen = Screen()
        self.running = False
        self.measuring = False
        self.reset()

    def reset(self):
        self.warmup_peaks = 0
        self.counting_peaks = 0
        self.peak_times = []
        self.last_peak_time = 0
        self.clean_samples = deque(maxlen=self.config["moving_avg_window"])
        self.frame_count = 0

    def run(self):
        self.running = True
        while self.running:
            self._detect()
            time.sleep(1.0 / 30)

    def _detect(self):
        x = self.config["detect_point"]["x"]
        y = self.config["detect_point"]["y"]
        size = self.config["sample_size"]

        current_b = self.screen.get_brightness(x, y, size)

        if not self.measuring:
            if len(self.clean_samples) > 0:
                avg = np.mean(list(self.clean_samples))
            else:
                avg = current_b
            self.brightness.emit(current_b, avg, False)
            self.clean_samples.append(current_b)
            return

        self.frame_count += 1

        if len(self.clean_samples) < 5:
            self.clean_samples.append(current_b)
            self.brightness.emit(current_b, current_b, False)
            return

        moving_avg = np.mean(list(self.clean_samples))
        threshold = moving_avg * (1 + self.config["threshold_pct"] / 100.0)

        is_peak = False
        now = time.time()

        if current_b > threshold:
            min_interval = self.config["min_peak_interval"]

            if now - self.last_peak_time > min_interval:
                is_peak = True
                self.last_peak_time = now

                warmup_needed = self.config["warmup_count"]
                slots_needed = self.config["slots_to_count"]

                if self.warmup_peaks < warmup_needed:
                    self.warmup_peaks += 1
                    self.status.emit(f"Warmup {self.warmup_peaks}/{warmup_needed}")
                    self.debug.emit(f"[Warmup] Peak {self.warmup_peaks}/{warmup_needed}")
                else:
                    self.counting_peaks += 1
                    self.peak_times.append(now)

                    points_needed = slots_needed + 1
                    self.status.emit(f"Counting {self.counting_peaks}/{points_needed}")
                    self.debug.emit(f"[Count] Peak {self.counting_peaks}/{points_needed}, B={current_b:.1f}")

                    if self.counting_peaks >= points_needed:
                        self._calculate_result()
        else:
            self.clean_samples.append(current_b)

        self.brightness.emit(current_b, moving_avg, is_peak)

    def _calculate_result(self):
        if len(self.peak_times) < 2:
            return

        time_span = self.peak_times[-1] - self.peak_times[0]
        slots = len(self.peak_times) - 1

        if time_span > 0:
            degrees_per_slot = 360.0 / 37.0
            total_degrees = slots * degrees_per_slot
            velocity = total_degrees / time_span
            self.result.emit(velocity, time_span, slots)
            self.debug.emit(f"RESULT: {velocity:.1f} deg/s, {slots} slots in {time_span:.3f}s")

        self.stop_measure()

    def start_measure(self):
        self.reset()
        self.measuring = True
        self.status.emit("Waiting for wheel to spin...")

    def stop_measure(self):
        self.measuring = False
        self.status.emit("Done")

    def stop(self):
        self.running = False
        self.wait()


# ==================== Hotkey ====================
class HotkeySignals(QObject):
    pressed = pyqtSignal(str)


class Hotkeys:
    def __init__(self, signals):
        self.signals = signals
        self.pressed = set()
        self.last = 0
        self.listener = None
        self._start_listener()

    def _start_listener(self):
        """启动键盘监听，带错误处理"""
        try:
            self.listener = keyboard.Listener(
                on_press=self._press,
                on_release=self._release
            )
            self.listener.start()
            print("[INFO] Keyboard listener started")
        except Exception as e:
            print(f"[WARN] Could not start keyboard listener: {e}")
            print("[WARN] Hotkeys will not work. Try running as administrator.")
            self.listener = None

    def _press(self, key):
        self.pressed.add(key)
        now = time.time()
        if now - self.last < 0.3:
            return

        # 检测 Ctrl 键（兼容 Windows 和 macOS）
        ctrl = any(k in self.pressed for k in [
            keyboard.Key.ctrl,
            keyboard.Key.ctrl_l,
            keyboard.Key.ctrl_r,
        ])

        if not ctrl:
            return

        for k in list(self.pressed):
            try:
                if hasattr(k, 'char') and k.char == '1':
                    self.last = now
                    self.signals.pressed.emit('set_point')
                    self.pressed.clear()
                    return
            except Exception:
                pass

    def _release(self, key):
        self.pressed.discard(key)

    def stop(self):
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass


# ==================== Main Window ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.history = []

        self.hk_signals = HotkeySignals()
        self.hk_signals.pressed.connect(self.on_hotkey)
        self.hotkeys = Hotkeys(self.hk_signals)

        self.detector = DetectorThread(self.config)
        self.detector.status.connect(self.on_status)
        self.detector.result.connect(self.on_result)
        self.detector.brightness.connect(self.on_brightness)
        self.detector.debug.connect(self.on_debug)
        self.detector.start()

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_mouse)
        self.timer.start(50)

    def init_ui(self):
        self.setWindowTitle("Roulette Speed v6.1")
        self.setFixedSize(380, 850)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.WindowCloseButtonHint)

        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # 标题 - 移除 emoji 以确保兼容性
        title = QLabel("Roulette Speed Detector v6.1")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2196F3;")
        layout.addWidget(title)

        # 平台信息
        platform_info = QLabel(f"Platform: {platform.system()}")
        platform_info.setAlignment(Qt.AlignCenter)
        platform_info.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(platform_info)

        info = QFrame()
        info.setStyleSheet("background: #e3f2fd; border-radius: 6px;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(2)

        self.mouse_label = QLabel("Mouse: (0, 0)")
        self.mouse_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #1565c0;")
        info_layout.addWidget(self.mouse_label)

        hint = QLabel("Ctrl+1: Set detection point")
        hint.setStyleSheet("font-size: 10px; color: #666;")
        info_layout.addWidget(hint)

        layout.addWidget(info)

        point_group = QGroupBox("Detection Point")
        point_layout = QVBoxLayout()
        point_layout.setSpacing(4)

        self.point_label = QLabel(f"({self.config['detect_point']['x']}, {self.config['detect_point']['y']})")
        self.point_label.setAlignment(Qt.AlignCenter)
        self.point_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        point_layout.addWidget(self.point_label)

        # 添加手动设置按钮（备用方案）
        self.set_point_btn = QPushButton("Set Point (or Ctrl+1)")
        self.set_point_btn.clicked.connect(lambda: self.on_hotkey('set_point'))
        self.set_point_btn.setStyleSheet("font-size: 11px;")
        point_layout.addWidget(self.set_point_btn)

        point_group.setLayout(point_layout)
        layout.addWidget(point_group)

        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sample Size (NxN):"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(3, 20)
        self.size_spin.setValue(self.config["sample_size"])
        self.size_spin.valueChanged.connect(self.on_size_change)
        row1.addWidget(self.size_spin)
        param_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Threshold (%):"))
        self.thresh_spin = QSpinBox()
        self.thresh_spin.setRange(1, 50)
        self.thresh_spin.setValue(int(self.config["threshold_pct"]))
        self.thresh_spin.valueChanged.connect(self.on_thresh_change)
        row2.addWidget(self.thresh_spin)
        param_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Moving Avg Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 100)
        self.window_spin.setValue(self.config["moving_avg_window"])
        self.window_spin.valueChanged.connect(self.on_window_change)
        row3.addWidget(self.window_spin)
        param_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Warmup Peaks:"))
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(1, 10)
        self.warmup_spin.setValue(self.config["warmup_count"])
        self.warmup_spin.valueChanged.connect(self.on_warmup_change)
        row4.addWidget(self.warmup_spin)
        param_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Slots to Count:"))
        self.slots_spin = QSpinBox()
        self.slots_spin.setRange(3, 20)
        self.slots_spin.setValue(self.config["slots_to_count"])
        self.slots_spin.valueChanged.connect(self.on_slots_change)
        row5.addWidget(self.slots_spin)
        param_layout.addLayout(row5)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        graph_group = QGroupBox("Brightness (green=current, red=avg, yellow=peak)")
        graph_layout = QVBoxLayout()

        self.graph = BrightnessGraph()
        graph_layout.addWidget(self.graph)

        self.bright_label = QLabel("B: -- | Avg: -- | Threshold: --")
        self.bright_label.setAlignment(Qt.AlignCenter)
        self.bright_label.setStyleSheet("font-size: 11px; color: #888; font-family: monospace;")
        graph_layout.addWidget(self.bright_label)

        graph_group.setLayout(graph_layout)
        layout.addWidget(graph_group)

        self.status_label = QLabel("Ready - Press START before wheel spins")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(40)
        self.status_label.setStyleSheet("""
            font-size: 13px; font-weight: bold;
            background: #333; color: #ffd700;
            border-radius: 6px;
        """)
        layout.addWidget(self.status_label)

        result_group = QGroupBox("Result")
        result_layout = QVBoxLayout()

        self.velocity_label = QLabel("-- deg/s")
        self.velocity_label.setAlignment(Qt.AlignCenter)
        self.velocity_label.setFixedHeight(55)
        self.velocity_label.setStyleSheet("""
            font-size: 32px; font-weight: bold; color: #00ff88;
            background: #1a1a2e; border-radius: 8px;
        """)
        result_layout.addWidget(self.velocity_label)

        self.detail_label = QLabel("Waiting...")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet("font-size: 11px; color: #aaa;")
        result_layout.addWidget(self.detail_label)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        history_group = QGroupBox("History (Last 6)")
        history_layout = QVBoxLayout()

        self.history_label = QLabel("No records")
        self.history_label.setFixedHeight(100)
        self.history_label.setAlignment(Qt.AlignTop)
        self.history_label.setStyleSheet("""
            font-size: 10px; font-family: monospace;
            background: #1a1a1a; color: #aaa;
            padding: 6px; border-radius: 4px;
        """)
        history_layout.addWidget(self.history_label)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        self.debug_label = QLabel("")
        self.debug_label.setFixedHeight(35)
        self.debug_label.setStyleSheet("""
            font-size: 9px; font-family: monospace;
            background: #0a0a0a; color: #666;
            padding: 4px; border-radius: 4px;
        """)
        layout.addWidget(self.debug_label)

        self.start_btn = QPushButton("START")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50; color: white;
                font-size: 18px; font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background: #45a049; }
        """)
        self.start_btn.clicked.connect(self.toggle)
        layout.addWidget(self.start_btn)

    def on_size_change(self, v):
        self.config["sample_size"] = v
        self.detector.config["sample_size"] = v
        save_config(self.config)

    def on_thresh_change(self, v):
        self.config["threshold_pct"] = float(v)
        self.detector.config["threshold_pct"] = float(v)
        save_config(self.config)

    def on_window_change(self, v):
        self.config["moving_avg_window"] = v
        self.detector.config["moving_avg_window"] = v
        save_config(self.config)

    def on_warmup_change(self, v):
        self.config["warmup_count"] = v
        self.detector.config["warmup_count"] = v
        save_config(self.config)

    def on_slots_change(self, v):
        self.config["slots_to_count"] = v
        self.detector.config["slots_to_count"] = v
        save_config(self.config)

    def update_mouse(self):
        pos = QCursor.pos()
        self.mouse_label.setText(f"Mouse: ({pos.x()}, {pos.y()})")

    def on_hotkey(self, action):
        if action == 'set_point':
            pos = QCursor.pos()
            self.config["detect_point"]["x"] = pos.x()
            self.config["detect_point"]["y"] = pos.y()
            self.detector.config["detect_point"]["x"] = pos.x()
            self.detector.config["detect_point"]["y"] = pos.y()
            save_config(self.config)
            self.point_label.setText(f"({pos.x()}, {pos.y()})")
            self.status_label.setText(f"Point set: ({pos.x()}, {pos.y()})")

    def on_status(self, text):
        self.status_label.setText(text)
        if "Waiting" in text:
            bg, color = "#37474f", "#b0bec5"
        elif "Warmup" in text:
            bg, color = "#e65100", "#ffe0b2"
        elif "Counting" in text:
            bg, color = "#1a237e", "#90caf9"
        elif "Done" in text:
            bg, color = "#2e7d32", "#c8e6c9"
        else:
            bg, color = "#333", "#ffd700"
        self.status_label.setStyleSheet(f"""
            font-size: 13px; font-weight: bold;
            background: {bg}; color: {color};
            border-radius: 6px;
        """)

    def on_result(self, velocity, time_span, slots):
        self.velocity_label.setText(f"{velocity:.1f} deg/s")
        cycle = 360 / velocity if velocity > 0 else 0
        self.detail_label.setText(f"{slots} slots in {time_span:.3f}s | Cycle: {cycle:.2f}s")

        t = time.strftime("%H:%M:%S")
        self.history.insert(0, f"{t}  {velocity:6.1f} deg/s  ({slots} slots, {time_span:.3f}s)")
        self.history = self.history[:6]
        self.history_label.setText("\n".join(self.history))

        self.start_btn.setText("START")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50; color: white;
                font-size: 18px; font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background: #45a049; }
        """)

    def on_brightness(self, b, avg, is_peak):
        threshold = avg * (1 + self.config["threshold_pct"] / 100.0) if avg > 0 else 0
        self.bright_label.setText(f"B: {b:.0f} | Avg: {avg:.0f} | Threshold: {threshold:.0f}")
        self.graph.add_point(b, avg, is_peak)

    def on_debug(self, msg):
        self.debug_label.setText(msg)
        print(msg)

    def toggle(self):
        if not self.detector.measuring:
            self.detector.start_measure()
            self.start_btn.setText("STOP")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: #f44336; color: white;
                    font-size: 18px; font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover { background: #d32f2f; }
            """)
        else:
            self.detector.stop_measure()
            self.start_btn.setText("START")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: #4CAF50; color: white;
                    font-size: 18px; font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover { background: #45a049; }
            """)

    def closeEvent(self, e):
        self.detector.stop()
        self.hotkeys.stop()
        e.accept()


def main():
    # Windows 控制台编码修复
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    print("=" * 50)
    print("  Roulette Speed Detector v6.1")
    print(f"  Platform: {platform.system()}")
    print("=" * 50)
    print("  1. Set detection point (Ctrl+1 or button)")
    print("  2. Click START before wheel spins")
    print("  3. Wait for auto-detection")
    print("=" * 50)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()