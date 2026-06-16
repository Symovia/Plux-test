"""PLUX acquisition with CSV save + real-time matplotlib plot.

Your setup:
  Address  BTH00:07:80:8C:AD:4F
  Port 1   Inductive Respiration (RIP)
  Port 2   Electrodermal Activity (EDA)
  Port 3   Electrodermal Activity (EDA)
  Rate     1000 Hz, 30 s, 16-bit, code 0x07 (ports 1+2+3)
"""
import csv
import platform
import sys
import time
import os
from collections import deque
from pathlib import Path

# Locate PLUX API: env var PLUX_API_DIR, or default to ../PLUX-API-Python3/Win64_X.X
_default_base = Path(__file__).resolve().parent.parent / "PLUX-API-Python3"
_major, _minor = platform.python_version_tuple()[:2]
_default_api_dir = _default_base / f"Win64_{_major}{_minor}"
_api_dir = Path(os.environ.get("PLUX_API_DIR", str(_default_api_dir)))
if not _api_dir.is_dir():
    print(f"ERROR: PLUX API not found at {_api_dir}.")
    print("Set env var PLUX_API_DIR to your plux.pyd directory.")
    sys.exit(1)
sys.path.append(str(_api_dir))
print(f"Loaded PLUX API: {_api_dir}")

import plux
import matplotlib.pyplot as plt

# Override with: $env:PLUX_ADDRESS = "BTH00:07:80:XX:XX:XX"
ADDRESS = os.environ.get("PLUX_ADDRESS", "BTH00:07:80:8C:AD:B3")
DURATION_SEC = 30
SAMPLE_RATE = 1000
RESOLUTION_BITS = 16
CODE = 0x07
CHANNEL_NAMES = ["RIP (port 1)", "ECG (port 2)", "EDA (port 3)"]

PLOT_WINDOW_SEC = 5
PLOT_UPDATE_EVERY = 50
BUFFER_LEN = PLOT_WINDOW_SEC * SAMPLE_RATE

OUTPUT_DIR = Path(__file__).resolve().parent / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)


class AcquisitionDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.MemoryDev.__init__(address)

    def setup(self, csv_path):
        self.target_samples = DURATION_SEC * SAMPLE_RATE
        self.n_ch = len(CHANNEL_NAMES)

        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["nSeq", "t_sec"] + CHANNEL_NAMES)

        self.t_buf = deque(maxlen=BUFFER_LEN)
        self.ch_buf = [deque(maxlen=BUFFER_LEN) for _ in range(self.n_ch)]

        plt.ion()
        fig, axes = plt.subplots(self.n_ch, 1, figsize=(11, 7), sharex=True)
        self.fig = fig
        self.axes = list(axes) if hasattr(axes, "__iter__") else [axes]
        self.lines = []
        for ax, name in zip(self.axes, CHANNEL_NAMES):
            (line,) = ax.plot([], [], lw=1)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            self.lines.append(line)
        self.axes[-1].set_xlabel("Time (s)")
        self.fig.suptitle(
            f"PLUX live - {SAMPLE_RATE} Hz, {DURATION_SEC}s, code=0x{CODE:02X}"
        )
        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw()

    def onRawFrame(self, nSeq, data):
        t = nSeq / SAMPLE_RATE
        self.csv_writer.writerow([nSeq, f"{t:.4f}"] + list(data))
        self.t_buf.append(t)
        for i in range(self.n_ch):
            self.ch_buf[i].append(data[i])

        if nSeq % PLOT_UPDATE_EVERY == 0:
            t_arr = list(self.t_buf)
            for line, buf in zip(self.lines, self.ch_buf):
                line.set_data(t_arr, list(buf))
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            if nSeq % SAMPLE_RATE == 0:
                self.csv_file.flush()
                print(f"  t={t:5.2f}s  data={tuple(data)}")

        return nSeq >= self.target_samples

    def close_io(self):
        self.csv_file.close()


def print_sensors(device):
    try:
        sensors = device.getSensors()
        print("\nDetected sensors:")
        for port, sensor in sensors.items():
            print(
                f"  port {port}: class={sensor.clas} serial={sensor.serialNum}"
            )
    except Exception as e:
        print(f"  (getSensors failed: {e})")


def main():
    csv_path = OUTPUT_DIR / f"plux_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Connecting to {ADDRESS} ...")
    device = AcquisitionDevice(ADDRESS)
    device.setup(csv_path)
    try:
        print(f"Battery: {int(device.getBattery())}%")
        print_sensors(device)
        print(f"\nStarting acquisition: {DURATION_SEC}s @ {SAMPLE_RATE} Hz")
        print(f"  Saving to: {csv_path}")
        device.start(SAMPLE_RATE, CODE, RESOLUTION_BITS)
        device.loop()
        device.stop()
        print("Acquisition complete.")
    except KeyboardInterrupt:
        print("\nInterrupted - saving partial data.")
        try:
            device.stop()
        except Exception:
            pass
    finally:
        device.close_io()
        try:
            device.close()
        except Exception:
            pass
        plt.ioff()
        print(f"\nCSV saved: {csv_path}")
        print("Close the plot window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
