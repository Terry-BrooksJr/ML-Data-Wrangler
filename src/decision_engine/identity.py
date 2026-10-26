"""Who is using the app: the computer login, as user@host.

Judgements (verdicts, outcomes, misclassifications) are recorded under it
rather than a typed-in name, so nobody can record one as someone else from
the app. It is only as trustworthy as the login: anyone signed in as that
user is that user.
"""

import getpass
import socket


def current_user() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # no login name (some containers)
        user = "unknown"
    return f"{user}@{socket.gethostname().split('.')[0]}"
