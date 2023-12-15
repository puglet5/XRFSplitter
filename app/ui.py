import json
import logging
import subprocess
import tkinter
from tkinter import END, NORMAL

from usbmonitor import USBMonitor
from usbmonitor.attributes import ID_MODEL, ID_MODEL_ID, ID_VENDOR_ID

logger = logging.getLogger(__name__)


def get_device_mountpoint():
    ...


def mount_device():
    ...


class UserInterface:
    def __init__(self):
        self.app = tkinter.Tk()
        self.app.title("XRF Splitter")
        self.app.protocol("WM_DELETE_WINDOW", self.quit)
        self.app.bind("<Control-q>", lambda _: self.quit())
        self.app.geometry("1280x720")
        self.monitor = USBMonitor()

        self.device_label = tkinter.Text(self.app, height=20, width=52)
        self.device_label.config(state=NORMAL)
        self.device_label.pack()

    def update_device_label(self, device_id, device_info):
        self.device_label.delete(1.0, END)
        p_handler = subprocess.run(
            ["lsblk", "-J", "-o", "PATH,SERIAL,MOUNTPOINT"],
            check=True,
            capture_output=True,
        )
        json_output = json.loads(p_handler.stdout.decode("utf-8"))
        # (serial, device_path, mount_point)
        drives = [
            (dev["serial"], dev["path"], dev["mountpoint"])
            for dev in json_output["blockdevices"]
        ]
        self.device_label.insert(
            END,
            str(drives),
        )

    def start(self):
        self.monitor.start_monitoring(
            on_connect=self.update_device_label, check_every_seconds=0.1
        )
        self.app.mainloop()

    def quit(self):
        self.app.quit()
