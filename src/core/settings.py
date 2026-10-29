"""User settings: the environment variables the app needs, editable from the UI.

Settings live in a per-user JSON file (settings_path()), so a packaged build
needs no shell setup. load() copies them into os.environ before the rest of
the app imports; save() writes the file and updates os.environ. Code reads
each setting from os.environ when it uses it, never into a constant at
import, so every change applies without restarting the app.
"""

import json
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

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
    """

    name: str
    label: str
    group: str
    secret: bool = False
    required: bool = False
    default: str = ""
    help: str = ""


SETTINGS: List[Setting] = [
    Setting(
        "MONGO_URI",
        "Mongo URI",
        "MongoDB",
        secret=True,
        required=True,
        help="Atlas (or mongodb-atlas-local) connection string; full-text search needs Atlas.",
    ),
    Setting("MONGO_DB", "Database", "MongoDB", default="support_db"),
    Setting("MONGO_COLLECTION", "Tickets collection", "MongoDB", default="tickets"),
    Setting(
        "CHEAPER_INFERENCE_API_KEY",
        "API key",
        "Decision Engine (Cheaper Inference)",
        secret=True,
        required=True,
    ),
    Setting(
        "CHEAPER_INFERENCE_API_URL",
        "API URL",
        "Decision Engine (Cheaper Inference)",
        required=True,
        default="https://api.cheaperinference.com",
        help="Must be https:// (plain http:// is allowed only for localhost): the API key and ticket text are sent here.",
    ),
    Setting(
        "CHEAPER_INFERENCE_PRIMARY_MODEL_NAME",
        "Primary model",
        "Decision Engine (Cheaper Inference)",
        required=True,
        help="Classifies each ticket.",
    ),
    Setting(
        "CHEAPER_INFERENCE_SECONDARY_MODEL_NAME",
        "Secondary model",
        "Decision Engine (Cheaper Inference)",
        help="Fallback if the primary call errors. Leave blank to skip.",
    ),
    Setting(
        "CHEAPER_INFERENCE_CONFIRMATION_MODEL_NAME",
        "Confirmation model",
        "Decision Engine (Cheaper Inference)",
        help="Re-checks manual-triage and low-confidence results. Leave blank to skip.",
    ),
    Setting(
        "CHEAPER_INFERENCE_ARBITER_MODEL_NAME",
        "Arbiter model",
        "Decision Engine (Cheaper Inference)",
        help="Rules when the confirmation model disagrees with the first answer. "
        "Leave blank to send disagreements to manual triage.",
    ),
    Setting(
        "DECISION_MODELS",
        "Decision models",
        "Decision Engine (Cheaper Inference)",
        help=(
            "Comma-separated models to treat as decision models (classification only, "
            "A/B tests only) even if Cheaper Inference's model list can't be read. "
            "Usually blank: the list identifies them."
        ),
    ),
    Setting(
        "DECISION_ENGINE_MAX_TOKENS",
        "Max tokens",
        "Decision Engine (Cheaper Inference)",
        default="16000",
        help="The reply limit for every model call, thinking included, up to 128000. A "
        "reply cut off here is an engine error. Only the tokens used are billed.",
    ),
    Setting(
        "DECISION_ENGINE_EFFORT",
        "Effort",
        "Decision Engine (Cheaper Inference)",
        help="low, medium, high, xhigh or max; blank uses each model's default. Sent to "
        "every model, and only Claude models accept it: leave blank if any model isn't Claude.",
    ),
    Setting(
        "DECISION_ENGINE_TIMEOUT_SECONDS",
        "Request timeout (seconds)",
        "Decision Engine (Cheaper Inference)",
        default="120",
        help="How long one model call may take before it is retried, up to 3600.",
    ),
    Setting(
        "TAG_NORMALIZER_MODEL",
        "Tag Normalizer model",
        "Decision Engine (Cheaper Inference)",
        help="Used by the Tag Normalizer's language-model option. Leave blank to use the primary model.",
    ),
    Setting(
        "DECISION_ENGINE_MODEL",
        "Default model",
        "Decision Engine (Cheaper Inference)",
        default="claude-opus-4.8",
        help="Used when no primary model is set.",
    ),
    Setting(
        "MODEL_BUCKET",
        "Bucket",
        "Model storage (S3 / Backblaze B2)",
        help="Trained topic models are uploaded here after each run. Leave blank to keep them on this computer only.",
    ),
    Setting(
        "B2_KEY_ID",
        "Key ID",
        "Model storage (S3 / Backblaze B2)",
        help="Backblaze application key ID, or an AWS access key ID.",
    ),
    Setting(
        "B2_APPLICATION_KEY",
        "Secret key",
        "Model storage (S3 / Backblaze B2)",
        secret=True,
        help="Backblaze application key, or an AWS secret access key.",
    ),
    Setting(
        "MODEL_SIGNING_KEY",
        "Signing key",
        "Model storage (S3 / Backblaze B2)",
        secret=True,
        help="Needed with a bucket. Signs uploaded models and verifies downloaded ones, which are only "
        "opened if the signature matches. Use the same key on every computer that shares models; "
        "generate one with: python -m core.model_storage new-key",
    ),
    Setting(
        "B2_ENDPOINT_URL",
        "Endpoint URL",
        "Model storage (S3 / Backblaze B2)",
        help="e.g. https://s3.us-west-004.backblazeb2.com. Leave blank for AWS S3. Must be https://.",
    ),
    Setting(
        "MODEL_STORAGE_REGION",
        "Region",
        "Model storage (S3 / Backblaze B2)",
        help="Optional. Worked out from a Backblaze endpoint; for AWS, the bucket's region (e.g. us-east-1).",
    ),
    Setting(
        "MODEL_PREFIX",
        "Folder in bucket",
        "Model storage (S3 / Backblaze B2)",
        default="topic-models/",
    ),
    Setting(
        "EMBEDDING_API_KEY",
        "Voyage API key",
        "Embeddings (Voyage AI)",
        secret=True,
        help="Needed for Embed Tickets. Sent as a Bearer token to the endpoint below.",
    ),
    Setting(
        "EMBEDDING_API_URL",
        "Endpoint",
        "Embeddings (Voyage AI)",
        default="https://ai.mongodb.com/v1/embeddings",
        help="Ends in /embeddings. Must be https:// (plain http:// is allowed only for localhost): the API key and ticket text are sent here.",
    ),
    Setting("EMBEDDING_MODEL", "Model", "Embeddings (Voyage AI)", default="voyage-3.5"),
    Setting(
        "EMBEDDING_TPM",
        "Tokens per minute",
        "Embeddings (Voyage AI)",
        default="10000",
        help="Your key's Voyage rate limit; requests are paced to stay under it. "
        "10,000 without billing on Atlas; raise it once your limit is higher.",
    ),
    Setting(
        "EMBEDDING_RPM",
        "Requests per minute",
        "Embeddings (Voyage AI)",
        default="3",
        help="Your key's Voyage request limit (3 without billing on Atlas).",
    ),
    Setting(
        "EMBEDDING_DIMENSIONS",
        "Dimensions",
        "Embeddings (Voyage AI)",
        default="1024",
        help="256, 512, 1024 or 2048. Changing it re-embeds every ticket and needs the "
        "tickets_vector index dropped and recreated.",
    ),
    Setting(
        "TOPIC_MODEL_DIR",
        "Topic model folder",
        "Storage",
        default="topic_models",
        help="Where saved topic-model runs go. Relative paths are under the working folder.",
    ),
    Setting(
        "TOPIC_RUNS_COLLECTION",
        "LDA runs collection",
        "Storage",
        default="topic_modeling_run_log",
    ),
    Setting(
        "BERTOPIC_RUNS_COLLECTION",
        "BERTopic runs collection",
        "Storage",
        default="bertopic_run_log",
    ),
    Setting(
        "TAG_RUNS_COLLECTION",
        "Tag Normalizer runs collection",
        "Storage",
        default="tag_normalization_run_log",
    ),
    Setting(
        "NORMALIZED_TAGS_COLLECTION",
        "Normalized tags collection",
        "Storage",
        default="normalized_tags",
        help="One document per successor tag, listing the tags it replaces.",
    ),
    Setting(
        "MISCLASSIFICATION_COLLECTION",
        "Misclassification collection",
        "Storage",
        default="misclassification_log",
    ),
    Setting(
        "LESSONS_COLLECTION",
        "Lessons collection",
        "Storage",
        default="lessons_learned",
    ),
    Setting(
        "AB_TEST_COLLECTION",
        "A/B runs collection",
        "Storage",
        default="ab_tests",
        help="One document per A/B test: its two engine configurations.",
    ),
    Setting(
        "AB_RESULT_COLLECTION",
        "A/B results collection",
        "Storage",
        default="ab_results",
        help="One document per ticket in an A/B test: both answers and the verdict.",
    ),
]


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def config_dir() -> pathlib.Path:
    """The per-user folder that holds the settings file."""
    if sys.platform == "darwin":
        base = pathlib.Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = pathlib.Path(
            os.environ.get("APPDATA") or pathlib.Path.home() / "AppData" / "Roaming"
        )
    else:
        base = pathlib.Path(
            os.environ.get("XDG_CONFIG_HOME") or pathlib.Path.home() / ".config"
        )
    return base / APP_NAME


def settings_path() -> pathlib.Path:
    """The per-user JSON file that the settings dialog reads and writes.

    JSON rather than .env: python-dotenv does not round-trip every character
    (a quote in a password, say).
    """
    return config_dir() / "settings.json"


# Set by load() when the settings file couldn't be read; the app shows it.
load_error: Optional[str] = None


def stored() -> Dict[str, Optional[str]]:
    """The values saved in settings_path(), or {} if there is no file yet.

    A None value is a setting the user blanked: load() removes it from the
    environment, so a value in .env or the shell doesn't come back.

    Raises:
        OSError, ValueError: The file can't be read or isn't a JSON object
            (JSONDecodeError and UnicodeDecodeError are ValueErrors).
    """
    path = settings_path()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return {k: None if v is None else str(v) for k, v in data.items()}


def _set_aside(path: pathlib.Path, error: Exception) -> str:
    """Moves an unreadable settings file out of the way, keeping it for recovery.

    Returns:
        A message for the user saying what happened and where the file went.
    """
    backup = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d-%H%M%S}")
    try:
        path.replace(backup)
        kept = f"It was moved to {backup}."
    except OSError:
        kept = "Saving settings will replace it."
    return f"Could not read the settings file {path} ({error}). {kept} Re-enter your settings."


def workspace_dir() -> pathlib.Path:
    """Where a packaged build writes logs, wrangled data and saved runs.

    The app writes to relative paths; inside a .app bundle the working
    directory is "/", which is not writable.
    """
    return pathlib.Path.home() / "Documents" / APP_NAME


def load() -> None:
    """Loads settings into os.environ, before anything uses them.

    A project .env (when running from source) is loaded first, then the user
    settings file, so values saved from the dialog win, and settings blanked
    in the dialog are removed. Both override the shell, so a stale exported
    MONGO_URI can't silently replace them.

    An unreadable settings file is set aside and reported in load_error
    rather than stopping the app, so the user can re-enter settings.
    """
    global load_error
    load_error = None
    if not is_frozen():
        load_dotenv(override=True)
    try:
        values = stored()
    except (OSError, ValueError) as e:
        load_error = _set_aside(settings_path(), e)
        values = {}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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


# What stops working when a group's required settings are missing.
_MISSING_EFFECTS = {
    "MongoDB": "Write Tickets to MongoDB, recording topic runs in MongoDB, and the Decision Engine",
    "Decision Engine (Cheaper Inference)": "the Decision Engine (classifying tickets)",
}


def describe_missing(missing: List[str]) -> str:
    """Says what each group of missing required settings disables.

    Only the features that need a group's settings are named, so a missing
    model setting doesn't claim Write Tickets to MongoDB is unavailable.
    """
    by_name = {s.name: s for s in SETTINGS}
    lines = []
    for group in dict.fromkeys(by_name[name].group for name in missing):
        names = [name for name in missing if by_name[name].group == group]
        effect = _MISSING_EFFECTS.get(group, f"features that use the {group} settings")
        lines.append(f"Not set: {', '.join(names)}.\nUnavailable until set: {effect}.")
    return "\n\n".join(lines)


# Owner and SYSTEM only, not inherited from the parent folder ("P"); OICI
# passes it on to files created in the folder.
_OWNER_ONLY_SDDL = "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)"


def _restrict_to_owner(path: pathlib.Path) -> None:
    """On Windows, replaces path's ACL with owner-only access.

    Mode bits don't protect files on Windows: os.open's 0600 and chmod only
    toggle read-only, so a file would otherwise inherit whatever ACL its
    folder has. No-op elsewhere, where mkstemp's 0600 already applies.

    Raises:
        OSError: Windows refused the change.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    advapi32.SetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    dacl_security_information = 0x00000004
    protected_dacl_security_information = 0x80000000

    descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        _OWNER_ONLY_SDDL, 1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not advapi32.SetFileSecurityW(
            str(path),
            dacl_security_information | protected_dacl_security_information,
            descriptor,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.LocalFree(descriptor)


def _write_atomically(path: pathlib.Path, text: str) -> None:
    """Replaces path with text, leaving the old file intact if writing fails.

    Writes a temporary file in the same directory, then renames it over path,
    so a full disk or a crash can't leave a truncated settings file that
    would stop the next launch. The file holds API keys and the Mongo
    password, so it is owner-only before anything is written to it: mkstemp
    creates it 0600, and on Windows _restrict_to_owner sets its ACL (the
    rename keeps it). The folder is restricted too.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _restrict_to_owner(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            _restrict_to_owner(pathlib.Path(tmp))
        except BaseException:
            # Windows can't delete a file that is still open.
            os.close(fd)
            raise
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def save(values: Dict[str, str]) -> None:
    """Writes settings to the user file and applies them to os.environ.

    An empty value removes the setting, so the code's default applies. If
    .env or the shell would set it again on the next launch (or it was
    blanked before), the file records it as null so load() removes it.

    Args:
        values: Setting name to value, as entered in the dialog.
    """
    try:
        previous = stored()
    except (OSError, ValueError):
        previous = {}
    saved: Dict[str, Optional[str]] = {}
    for setting in SETTINGS:
        value = values.get(setting.name, "").strip()
        if value:
            saved[setting.name] = value
        elif os.environ.get(setting.name) or (
            setting.name in previous and previous[setting.name] is None
        ):
            saved[setting.name] = None
    _write_atomically(settings_path(), json.dumps(saved, indent=2))
    for setting in SETTINGS:
        if saved.get(setting.name):
            os.environ[setting.name] = saved[setting.name]
        else:
            os.environ.pop(setting.name, None)
