"""Spending PIN, roles and an audit log for paid API calls.

Every request that costs money (Cheaper Inference, Voyage AI, OpenAI) runs
only under a grant, and a grant needs the spending PIN. The PIN is handed
over the way Jenkins hands over its initial admin password: Lasso writes it
to a file in a secrets folder that only the computer's user can read, and
whoever starts a paid job reads it from there. Being able to read that file
is the proof of authority, so the folder can be a mounted volume (a tmpfs,
a Docker or Kubernetes secret) that an administrator controls.

- **Secrets folder:** LASSO_SECRETS_DIR, else secrets/ in the settings
  folder. Created owner-only (0700). A folder or file that other users can
  write to is refused.
- **PIN file:** spend_pin, owner-only (0600), 8 digits. Lasso replaces it
  every pin_ttl_hours (default 12) and after MAX_BAD_PINS wrong tries, which
  also lock paid calls for LOCKOUT_MINUTES. A PIN file in a folder Lasso
  can't write to (a read-only mount) is used as provided and never replaced.
- **Grants:** the app asks for the PIN before each paid job; the grant covers
  that job's thread only, and only the services the job named. A command
  line run asks once on its terminal, or reads the PIN from the file named by
  LASSO_SPEND_PIN_FILE (for scheduled runs that mount the secrets folder).
- **Roles:** access.json in the secrets folder maps logins (user@host, or
  user) to viewer, reviewer, operator or admin; see PERMISSIONS. Without the
  file every login is admin (one person, one computer). It can also set
  daily_paid_call_limit, a cap on paid requests per day on this computer.
- **Audit log:** audit.log in the settings folder (LASSO_AUDIT_LOG), JSON
  lines: every PIN attempt, grant, paid request and refusal. The PIN itself
  is never logged.

Roles and the PIN only bind someone who can't edit the secrets folder or
Lasso's code. On a shared computer, the folder should belong to an
administrator's account, or be mounted read-only.

    cd src && python -m spend_guard where     # the PIN file's path
    cd src && python -m spend_guard rotate    # replace the PIN now (admin)
    cd src && python -m spend_guard whoami    # login, role and permissions
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import json
import os
import pathlib
import secrets
import socket
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional

import settings
from loguru import logger

SECRETS_DIR_ENV = "LASSO_SECRETS_DIR"
AUDIT_LOG_ENV = "LASSO_AUDIT_LOG"
PIN_FILE_ENV = "LASSO_SPEND_PIN_FILE"

PIN_FILE = "spend_pin"
STATE_FILE = "spend_state.json"
ACCESS_FILE = "access.json"
AUDIT_FILE = "audit.log"

PIN_DIGITS = 8
DEFAULT_PIN_TTL_HOURS = 12.0
MAX_BAD_PINS = 5
LOCKOUT_MINUTES = 15
AUDIT_MAX_BYTES = 5_000_000

# Services that bill per request.
CHEAPER_INFERENCE = "Cheaper Inference"
VOYAGE = "Voyage AI"
OPENAI = "OpenAI"
ALL_SERVICES: FrozenSet[str] = frozenset({CHEAPER_INFERENCE, VOYAGE, OPENAI})

# Permissions.
REVIEW = "review"  # verdicts, feedback, weekly review
SPEND = "spend"  # paid API calls
WRITE_DATA = "write_data"  # writes tickets and runs to MongoDB / S3
CHANGE_SETTINGS = "change_settings"
MANAGE_SECURITY = "manage_security"  # rotate the PIN

ROLES: Dict[str, FrozenSet[str]] = {
    "viewer": frozenset(),
    "reviewer": frozenset({REVIEW}),
    "operator": frozenset({REVIEW, SPEND, WRITE_DATA}),
    "admin": frozenset({REVIEW, SPEND, WRITE_DATA, CHANGE_SETTINGS, MANAGE_SECURITY}),
}


class SecurityError(Exception):
    """Base class: the action was refused. The message is shown to the user."""


class InsecureSecretsDir(SecurityError):
    """The secrets folder or a file in it can be written by other users."""


class PermissionDenied(SecurityError):
    """This login's role doesn't allow the action."""


class SpendNotAuthorized(SecurityError):
    """A paid call was made outside a grant for its service."""


class LockedOut(SecurityError):
    """Too many wrong PINs; paid calls are locked for a while."""


class WrongPin(SecurityError):
    def __init__(self, remaining: int):
        self.remaining = remaining
        super().__init__(
            f"Wrong PIN. {remaining} more tr{'y' if remaining == 1 else 'ies'} "
            f"before paid calls are locked for {LOCKOUT_MINUTES} minutes."
        )


class DailyLimitReached(SecurityError):
    """daily_paid_call_limit in access.json was reached."""


# --- identity and roles -----------------------------------------------------------


def identity() -> str:
    """This computer login as user@host (as A/B verdicts record it)."""
    try:
        user = getpass.getuser()
    except Exception:  # no login name (some containers)
        user = "unknown"
    return f"{user}@{socket.gethostname().split('.')[0]}"


def secrets_dir() -> pathlib.Path:
    configured = os.environ.get(SECRETS_DIR_ENV, "").strip()
    return (
        pathlib.Path(configured).expanduser()
        if configured
        else settings.config_dir() / "secrets"
    )


def audit_path() -> pathlib.Path:
    configured = os.environ.get(AUDIT_LOG_ENV, "").strip()
    return (
        pathlib.Path(configured).expanduser()
        if configured
        else settings.config_dir() / AUDIT_FILE
    )


def _check_not_writable_by_others(path: pathlib.Path) -> None:
    """Refuses a path that users other than its owner could write to.

    The owner may be another account (an administrator's) or root, as with a
    mounted secret: that is how roles are made binding.

    Raises:
        InsecureSecretsDir: group- or other-writable.
    """
    if sys.platform == "win32":  # mode bits mean little there; ACLs are set on write
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise InsecureSecretsDir(
            f"{path} can be changed by other users (mode {stat.filemode(mode)}). "
            f"Run: chmod go-w {path}"
        )


def ensure_secrets_dir() -> pathlib.Path:
    """The secrets folder, created owner-only if missing.

    Raises:
        InsecureSecretsDir: see _check_not_writable_by_others.
    """
    folder = secrets_dir()
    if not folder.exists():
        folder.mkdir(parents=True, mode=0o700)
        settings._restrict_to_owner(folder)
    _check_not_writable_by_others(folder)
    return folder


@dataclass(frozen=True)
class Policy:
    """access.json, or the single-user defaults when it doesn't exist."""

    users: Dict[str, str] = field(default_factory=dict)
    default_role: str = "admin"
    daily_paid_call_limit: Optional[int] = None
    pin_ttl_hours: float = DEFAULT_PIN_TTL_HOURS
    enforced: bool = False  # False: no access.json, everyone is admin

    def role_of(self, who: str) -> str:
        user = who.split("@", 1)[0]
        return self.users.get(who) or self.users.get(user) or self.default_role


def load_policy() -> Policy:
    """Reads access.json from the secrets folder.

    A file that can't be read or doesn't parse is treated as granting
    nothing (every login a viewer): a broken policy must not open access.
    """
    path = secrets_dir() / ACCESS_FILE
    if not path.exists():
        return Policy()
    try:
        _check_not_writable_by_others(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        users = data.get("users") or {}
        default_role = data.get("default_role", "viewer")
        for role in [default_role, *users.values()]:
            if role not in ROLES:
                raise ValueError(
                    f"unknown role {role!r}; use one of {', '.join(ROLES)}"
                )
        limit = data.get("daily_paid_call_limit")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("daily_paid_call_limit must be a whole number, 0 or more")
        ttl = float(data.get("pin_ttl_hours", DEFAULT_PIN_TTL_HOURS))
        if ttl <= 0:
            raise ValueError("pin_ttl_hours must be more than 0")
        return Policy(
            users={str(k): str(v) for k, v in users.items()},
            default_role=default_role,
            daily_paid_call_limit=limit,
            pin_ttl_hours=ttl,
            enforced=True,
        )
    except (OSError, ValueError, TypeError, SecurityError) as e:
        logger.error(f"{path} is unusable, so every login is a viewer: {e}")
        return Policy(default_role="viewer", enforced=True)


def role(who: Optional[str] = None) -> str:
    return load_policy().role_of(who or identity())


def permissions(who: Optional[str] = None) -> FrozenSet[str]:
    return ROLES[role(who)]


def can(permission: str, who: Optional[str] = None) -> bool:
    return permission in permissions(who)


def require_permission(permission: str, action: str) -> None:
    """Raises PermissionDenied (and audits it) unless this login may do permission.

    Args:
        action: What was attempted, in words, for the message and the log.
    """
    who = identity()
    if can(permission, who):
        return
    audit("denied", action=action, permission=permission)
    raise PermissionDenied(
        f"{who} is a {role(who)} and can't {action}. "
        f"Ask an admin to change your role in {secrets_dir() / ACCESS_FILE}."
    )


# --- audit log --------------------------------------------------------------------

_audit_lock = threading.Lock()


def audit(event: str, **details: Any) -> None:
    """Appends one JSON line to the audit log. Never raises: a full disk
    mustn't stop work, but it is logged."""
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": identity(),
        "pid": os.getpid(),
        "event": event,
        **details,
    }
    path = audit_path()
    try:
        with _audit_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > AUDIT_MAX_BYTES:
                os.replace(path, path.with_name(path.name + ".1"))
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
    except OSError as e:
        logger.error(f"Couldn't write the audit log {path}: {e}")


# --- the PIN ----------------------------------------------------------------------

_pin_lock = threading.RLock()


def pin_path() -> pathlib.Path:
    return secrets_dir() / PIN_FILE


def _writable(folder: pathlib.Path) -> bool:
    return os.access(folder, os.W_OK)


def _new_pin() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(PIN_DIGITS))


def rotate_pin(reason: str = "rotated") -> pathlib.Path:
    """Writes a new PIN to the PIN file (owner-only) and returns its path.

    Raises:
        InsecureSecretsDir: see ensure_secrets_dir.
        OSError: the folder is read-only (a provided PIN can't be replaced).
    """
    with _pin_lock:
        folder = ensure_secrets_dir()
        path = folder / PIN_FILE
        settings._write_atomically(path, _new_pin() + "\n")
        if sys.platform != "win32":
            os.chmod(path, 0o600)
        audit("pin_rotated", reason=reason, path=str(path))
        logger.info(
            f"A new spending PIN is in {path}. Paid calls (model and embedding "
            "requests) ask for it."
        )
        return path


def ensure_pin() -> pathlib.Path:
    """The PIN file's path, writing a PIN first if there is none or it expired.

    Raises:
        InsecureSecretsDir: the folder or PIN file can be written by others.
        SecurityError: no PIN file, and the folder is read-only.
    """
    with _pin_lock:
        folder = ensure_secrets_dir()
        path = folder / PIN_FILE
        if not path.exists():
            if not _writable(folder):
                raise SecurityError(
                    f"{folder} is read-only and holds no {PIN_FILE}: provide one there."
                )
            return rotate_pin("first use")
        _check_not_writable_by_others(path)
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > load_policy().pin_ttl_hours and _writable(folder):
            return rotate_pin("expired")
        return path


def _read_pin(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# Used instead of the state file when the secrets folder is read-only.
_memory_state: dict = {}


def _state() -> dict:
    """Wrong-PIN count, lockout and daily call counts, shared by every process."""
    try:
        stored = json.loads((secrets_dir() / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    return {**(stored if isinstance(stored, dict) else {}), **_memory_state}


def _update_state(**changes: Any) -> None:
    """Merges changes into the state; a None value removes that key."""
    state = _state()
    state.update(changes)
    state = {k: v for k, v in state.items() if v is not None}
    folder = secrets_dir()
    if _writable(folder):
        settings._write_atomically(folder / STATE_FILE, json.dumps(state))
    else:
        _memory_state.clear()
        _memory_state.update(state)


def locked_until() -> Optional[float]:
    """When the lockout ends (epoch seconds), or None if not locked."""
    until = _state().get("locked_until")
    return until if until and until > time.time() else None


def check_pin(entered: str, operation: str = "") -> None:
    """Checks a PIN against the PIN file.

    Wrong PINs are counted across processes (in the secrets folder); the
    MAX_BAD_PINS-th locks paid calls for LOCKOUT_MINUTES and replaces the
    PIN, so guesses made before the lockout are worthless after it.

    Raises:
        LockedOut, WrongPin, InsecureSecretsDir, SecurityError.
    """
    with _pin_lock:
        until = locked_until()
        if until:
            audit("pin_refused_locked", operation=operation)
            raise LockedOut(
                f"Too many wrong PINs. Paid calls are locked until "
                f"{datetime.fromtimestamp(until).strftime('%H:%M')}."
            )
        path = ensure_pin()
        expected = _read_pin(path)
        if expected and hmac.compare_digest(
            entered.strip().encode(), expected.encode()
        ):
            _update_state(bad=None, locked_until=None)
            audit("pin_accepted", operation=operation)
            return
        bad = int(_state().get("bad", 0)) + 1
        if bad >= MAX_BAD_PINS:
            _update_state(bad=None, locked_until=time.time() + LOCKOUT_MINUTES * 60)
            audit("pin_rejected_locked", operation=operation, attempts=bad)
            if _writable(path.parent):
                rotate_pin("too many wrong PINs")
            raise LockedOut(
                f"Too many wrong PINs. Paid calls are locked for {LOCKOUT_MINUTES} "
                "minutes, and the PIN has been replaced."
            )
        _update_state(bad=bad)
        audit("pin_rejected", operation=operation, attempts=bad)
        raise WrongPin(MAX_BAD_PINS - bad)


# --- grants -----------------------------------------------------------------------


@dataclass
class Grant:
    """Permission for one job to make paid calls to the named services.

    Active only on the thread running it (run/wrap/with), so a grant for one
    button never pays for another job.
    """

    operation: str
    services: FrozenSet[str]
    user: str = field(default_factory=identity)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    calls: int = 0

    def __enter__(self) -> "Grant":
        _active().append(self)
        audit(
            "grant_started",
            grant=self.id,
            operation=self.operation,
            services=sorted(self.services),
        )
        return self

    def __exit__(self, *exc) -> None:
        stack = _active()
        if self in stack:
            stack.remove(self)
        audit("grant_ended", grant=self.id, operation=self.operation, calls=self.calls)

    def run(self, task: Callable[[], Any]) -> Any:
        with self:
            return task()

    def wrap(self, task: Callable[[], Any]) -> Callable[[], Any]:
        """task, made to run under this grant (for handing to a worker thread)."""
        return lambda: self.run(task)


_local = threading.local()
_cli_grant: Optional[Grant] = None
_cli_lock = threading.Lock()
_interactive_cli = True


def _active() -> List[Grant]:
    if not hasattr(_local, "grants"):
        _local.grants = []
    return _local.grants


def gui_mode() -> None:
    """Called by the app: paid calls never fall back to a terminal prompt."""
    global _interactive_cli
    _interactive_cli = False


def authorize(
    pin: str, operation: str, services: Iterable[str] = ALL_SERVICES
) -> Grant:
    """A grant for operation, after checking the role and the PIN.

    Raises:
        PermissionDenied, LockedOut, WrongPin, InsecureSecretsDir, SecurityError.
    """
    require_permission(SPEND, f"start paid calls ({operation})")
    check_pin(pin, operation)
    return Grant(operation, frozenset(services))


def describe(services: Iterable[str]) -> str:
    return ", ".join(sorted(services))


def _authorize_cli(service: str) -> Grant:
    """For command-line runs: the PIN from LASSO_SPEND_PIN_FILE, else the terminal.

    One grant covers the rest of the process, on every thread.
    """
    global _cli_grant
    with _cli_lock:
        if _cli_grant is not None:
            return _cli_grant
        operation = (
            "command line: "
            + " ".join([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]])[:200]
        )
        pin_file = os.environ.get(PIN_FILE_ENV, "").strip()
        if pin_file:
            pin = _read_pin(pathlib.Path(pin_file).expanduser())
        elif _interactive_cli and sys.stdin and sys.stdin.isatty():
            print(
                f"This run makes paid requests ({service}). "
                f"Enter the spending PIN from {ensure_pin()}",
                file=sys.stderr,
            )
            while True:
                try:
                    _cli_grant = authorize(getpass.getpass("PIN: "), operation)
                    break
                except WrongPin as e:
                    print(e, file=sys.stderr)
            audit("grant_started", grant=_cli_grant.id, operation=operation)
            return _cli_grant
        else:
            audit("denied", action="paid call without a grant", service=service)
            raise SpendNotAuthorized(
                f"Paid {service} calls need the spending PIN. Run from a terminal, "
                f"or set {PIN_FILE_ENV} to the PIN file ({pin_path()})."
            )
        _cli_grant = authorize(pin, operation)
        audit("grant_started", grant=_cli_grant.id, operation=operation)
        return _cli_grant


def authorize_cli(service: str = CHEAPER_INFERENCE) -> None:
    """Asks for the PIN up front (command-line entry points), not at the first call."""
    if _cli_grant is None and not any(service in g.services for g in _active()):
        _authorize_cli(service)


def require(service: str, what: str = "") -> None:
    """Call before each paid request. Counts and audits it.

    Raises:
        SpendNotAuthorized: no grant for service on this thread (and, in the
            app, no terminal fallback).
        DailyLimitReached: access.json's daily_paid_call_limit was reached.
        PermissionDenied, LockedOut, WrongPin: from the command-line prompt.
    """
    grant = next((g for g in reversed(_active()) if service in g.services), None)
    if grant is None:
        if not _interactive_cli:
            audit(
                "denied", action="paid call without a grant", service=service, what=what
            )
            raise SpendNotAuthorized(
                f"A paid {service} call was attempted without the spending PIN; nothing was sent."
            )
        grant = _authorize_cli(service)
    _count_today(service)
    grant.calls += 1
    audit(
        "paid_call",
        grant=grant.id,
        operation=grant.operation,
        service=service,
        what=what,
    )


_count_lock = threading.Lock()


def _count_today(service: str) -> None:
    """Counts paid requests per day on this computer, refusing past the limit.

    The count is kept in the secrets folder; several processes at once can
    each read the same count, so the limit can be overshot slightly.
    """
    limit = load_policy().daily_paid_call_limit
    if limit is None:
        return
    with _count_lock, _pin_lock:
        today = datetime.now(timezone.utc).date().isoformat()
        used = int(_state().get("calls", {}).get(today, 0))
        if used >= limit:
            audit(
                "denied",
                action="paid call over the daily limit",
                service=service,
                limit=limit,
            )
            raise DailyLimitReached(
                f"The daily limit of {limit} paid calls (access.json) was reached; "
                "nothing was sent."
            )
        _update_state(calls={today: used + 1})


# --- command line -----------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spend_guard", description=__doc__.split("\n\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "where", help="print the PIN file's path (writing a PIN if needed)"
    )
    commands.add_parser("rotate", help="replace the PIN now (admin)")
    commands.add_parser("whoami", help="this login, its role and permissions")
    args = parser.parse_args(argv)
    settings.load()
    try:
        if args.command == "where":
            print(ensure_pin())
        elif args.command == "rotate":
            require_permission(MANAGE_SECURITY, "rotate the spending PIN")
            print(rotate_pin("requested"))
        else:
            policy = load_policy()
            who = identity()
            print(
                f"{who}: {policy.role_of(who)} ({', '.join(sorted(ROLES[policy.role_of(who)])) or 'read only'})"
            )
            if not policy.enforced:
                print(f"No {secrets_dir() / ACCESS_FILE}: every login is admin.")
    except (SecurityError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
