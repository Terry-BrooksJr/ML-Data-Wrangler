"""User settings: the environment variables the app needs, editable from the UI.

Settings live in a per-user JSON file (settings_path()), so a packaged build
needs no shell setup. load() copies them into os.environ before the rest of
the app imports; save() writes the file and updates os.environ, so most
changes apply without a restart. A few values are read once at import time
(Setting.restart) and need one.
"""

import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv

APP_NAME = "Lasso"


@dataclass(frozen=True)
class Setting:
    """One environment variable shown in the settings dialog.

    Attributes:
        name: The environment variable.
        label: Field label in the dialog.
        group: Section heading the field is listed under.
        secret: True to mask the value while editing.
        required: True if some feature refuses to run without it.
        default: Placeholder text; what the code falls back to when unset.
        help: Tooltip text.
        restart: True if the value is read at import, so a change needs a restart.
    """

    name: str
    label: str
    group: str
    secret: bool = False
    required: bool = False
    default: str = ""
    help: str = ""
    restart: bool = False


SETTINGS: List[Setting] = [
    Setting("MONGO_URI", "Mongo URI", "MongoDB", secret=True, required=True,
            help="Atlas (or mongodb-atlas-local) connection string; full-text search needs Atlas."),
    Setting("MONGO_DB", "Database", "MongoDB", default="support_db"),
    Setting("MONGO_COLLECTION", "Tickets collection", "MongoDB", default="tickets"),
    Setting("CHEAPER_INFERENCE_API_KEY", "API key", "Decision Engine", secret=True, required=True),
    Setting("CHEAPER_INFERENCE_API_URL", "API URL", "Decision Engine", required=True,
            default="https://api.cheaperinference.com"),
    Setting("CHEAPER_INFERENCE_PRIMARY_MODEL_NAME", "Primary model", "Decision Engine", required=True,
            help="Classifies each ticket."),
    Setting("CHEAPER_INFERENCE_SECONDARY_MODEL_NAME", "Secondary model", "Decision Engine",
            help="Fallback if the primary call errors. Leave blank to skip."),
    Setting("CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME", "Confirmation model", "Decision Engine",
            help="Re-checks manual-triage and low-confidence results. Leave blank to skip."),
    Setting("DECISION_ENGINE_MODEL", "Default model", "Decision Engine", default="claude-opus-4.8",
            help="Used when no primary model is set.", restart=True),
    Setting("B2_KEY_ID", "Key ID", "Backblaze B2", required=True),
    Setting("B2_APPLICATION_KEY", "Application key", "Backblaze B2", secret=True, required=True),
    Setting("B2_ENDPOINT_URL", "Endpoint URL", "Backblaze B2", required=True),
    Setting("TOPIC_MODEL_DIR", "Topic model folder", "Storage", default="topic_models",
            help="Where saved topic-model runs go. Relative paths are under the working folder."),
    Setting("TOPIC_RUNS_COLLECTION", "Topic runs collection", "Storage", default="topic_model_runs", restart=True),
    Setting("MISCLASSIFICATION_COLLECTION", "Misclassification collection", "Storage",
            default="misclassification_log", restart=True),
    Setting("LESSONS_COLLECTION", "Lessons collection", "Storage", default="lessons_learned", restart=True),
]


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def config_dir() -> pathlib.Path:
    """The per-user folder that holds the settings file."""
    if sys.platform == "darwin":
        base = pathlib.Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = pathlib.Path(os.environ.get("APPDATA") or pathlib.Path.home() / "AppData" / "Roaming")
    else:
        base = pathlib.Path(os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config")
    return base / APP_NAME


def settings_path() -> pathlib.Path:
    """The per-user JSON file that the settings dialog reads and writes.

    JSON rather than .env: python-dotenv does not round-trip every character
    (a quote in a password, say).
    """
    return config_dir() / "settings.json"


def stored() -> Dict[str, str]:
    """The values saved in settings_path(), or {} if there is no file yet."""
    path = settings_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {k: str(v) for k, v in json.load(f).items()}


def workspace_dir() -> pathlib.Path:
    """Where a packaged build writes logs, wrangled data and saved runs.

    The app writes to relative paths; inside a .app bundle the working
    directory is "/", which is not writable.
    """
    return pathlib.Path.home() / "Documents" / APP_NAME


def load() -> None:
    """Loads settings into os.environ, before modules that read them at import.

    A project .env (when running from source) is loaded first, then the user
    settings file, so values saved from the dialog win. Both override the
    shell, so a stale exported MONGO_URI can't silently replace them.
    """
    if not is_frozen():
        load_dotenv(override=True)
    os.environ.update(stored())


def use_workspace() -> pathlib.Path:
    """In a packaged build, makes workspace_dir() the working directory."""
    workspace = workspace_dir()
    if is_frozen():
        workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(workspace)
    return workspace


def current() -> Dict[str, str]:
    """The current value of each setting (empty string when unset)."""
    return {s.name: os.environ.get(s.name, "") for s in SETTINGS}


def missing_required() -> List[str]:
    """Names of required settings that are unset or empty."""
    return [s.name for s in SETTINGS if s.required and not os.environ.get(s.name)]


def save(values: Dict[str, str]) -> List[str]:
    """Writes settings to the user file and applies them to os.environ.

    An empty value removes the setting, so the code's default applies.

    Args:
        values: Setting name to value, as entered in the dialog.

    Returns:
        Names of changed settings that only take effect after a restart.
    """
    saved = {}
    restart_needed = []
    for setting in SETTINGS:
        value = values.get(setting.name, "").strip()
        if setting.restart and value != os.environ.get(setting.name, ""):
            restart_needed.append(setting.name)
        if value:
            saved[setting.name] = value
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Holds API keys and the Mongo password, so create it owner-only.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    path.chmod(0o600)
    for setting in SETTINGS:
        if setting.name in saved:
            os.environ[setting.name] = saved[setting.name]
        else:
            os.environ.pop(setting.name, None)
    return restart_needed
