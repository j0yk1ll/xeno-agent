import logging
from pathlib import Path
from threading import RLock
from typing import Callable

import yaml


class SettingsManager:
    def __init__(self):
        self._lock = RLock()
        self.settings = {}
        self.callbacks = []

        read_path, write_path = self._determine_settings_paths()
        self.read_path = Path(read_path)
        self.write_path = Path(write_path)

        self._load_settings()

    def _determine_settings_paths(self):
        """
        Determine .settings.yml paths (READ vs WRITE) and create .xeno folder if needed

        - settings_path_read: Path from which we'll load settings
        - settings_path_write: Path to which we'll save settings (always ~/.xeno/.settings.yml)
        """

        xeno_dir = Path.home() / ".xeno"
        xeno_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

        xeno_settings = xeno_dir / ".settings.yml"
        local_settings = Path.cwd() / ".settings.yml"

        # 1) If ~/.xeno/.settings.yml exists, prefer reading from there
        if xeno_settings.exists():
            settings_path_read = str(xeno_settings)
        # 2) Else if a local .settings.yml exists, use it to read from
        elif local_settings.exists():
            settings_path_read = str(local_settings)
        # 3) Else fallback to ~/.xeno/.settings.yml (even if it doesn't exist yet, we'll create later)
        else:
            settings_path_read = str(xeno_settings)

        settings_path_write = str(xeno_settings)

        logging.debug(f"settings_path_read = {settings_path_read}")
        logging.debug(f"settings_path_write = {settings_path_write}")

        return settings_path_read, settings_path_write

    def _load_settings(self):
        with self._lock:
            self.settings = self._load_from_file(self.read_path)

    def _load_from_file(self, path: Path) -> dict:
        try:
            with path.open("r") as f:
                data = yaml.safe_load(f) or {}
                return data
        except FileNotFoundError:
            logging.warning(f"{path} not found.")
            return {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing {path}: {e}")
            return {}

    def save_settings(self):
        with self._lock:
            try:
                with open(self.write_path, "w") as f:
                    yaml.safe_dump(self.settings, f)
                    logging.debug(f"Saved settings to {self.write_path}")
            except Exception as e:
                logging.error(f"Failed to save settings to {self.write_path}: {e}")

            # Notify listeners about update
            for callback in self.callbacks:
                callback()

    def get_settings(self):
        with self._lock:
            return self.settings

    def get_settings_key(self, key, default=None):
        with self._lock:
            return self.settings.get(key, default)

    def set_settings_key(self, key, value):
        with self._lock:
            self.settings[key] = value

    def on_update(self, callback: Callable):
        with self._lock:
            self.callbacks.append(callback)
