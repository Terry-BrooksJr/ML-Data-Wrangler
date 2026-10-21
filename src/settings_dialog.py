"""Settings dialog: edit the environment variables in settings.SETTINGS."""

from itertools import groupby
from typing import Dict
from urllib.parse import urlsplit

import settings
from decision_engine.classify import api_url_problem, is_loopback_url
from embeddings import ALLOWED_DIMENSIONS, parse_dimensions
from loguru import logger
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from widgets import Card, hint, LineEdit, use_si_scrollbars

# Width of the label part of each field, so a group's fields line up.
TITLE_WIDTH = 190


class SettingsDialog(QDialog):
    """Lists every setting by group; Save writes them via settings.save().

    Required fields are marked with *. Secret fields are masked, with an eye
    toggle to reveal them.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(680, 700)
        self.fields: Dict[str, QLineEdit] = {}

        intro = hint(
            "Fields marked * are needed by the Decision Engine (Mongo URI also by "
            "Write Tickets to MongoDB). Leave a field blank to use its default, even if "
            f".env or your shell sets it. Saved to:\n{settings.settings_path()}"
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(12)
        values = settings.current()
        for group, items in groupby(settings.SETTINGS, key=lambda s: s.group):
            box = Card(group)
            for setting in items:
                label = setting.label + (" *" if setting.required else "")
                field = LineEdit(label, values[setting.name])
                field.setTitleWidthMode(LineEdit.TitleWidthMode.Fixed)
                field.setTitleFixedWidth(TITLE_WIDTH)
                field.setPlaceholderText(setting.default)
                field.setClearButtonEnabled(True)
                tooltip = setting.name + (f"\n{setting.help}" if setting.help else "")
                if setting.restart:
                    tooltip += "\nChanges take effect after a restart."
                field.setToolTip(tooltip)
                if setting.secret:
                    self._make_secret(field)
                box.body.addWidget(field)
                self.fields[setting.name] = field
            content_layout.addWidget(box)
        content_layout.addStretch()

        scroll = use_si_scrollbars(QScrollArea())
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        content.setAutoFillBackground(False)
        scroll.viewport().setAutoFillBackground(False)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(intro)
        layout.addWidget(scroll)
        layout.addWidget(buttons)

    @staticmethod
    def _make_secret(field: QLineEdit) -> None:
        """Masks a field and adds a trailing toggle that reveals it."""
        field.setEchoMode(QLineEdit.Password)
        toggle = QAction("Show", field)
        toggle.setCheckable(True)
        toggle.setToolTip("Show / hide")
        toggle.setIcon(
            field.style().standardIcon(field.style().SP_FileDialogContentsView)
        )
        toggle.toggled.connect(
            lambda shown: field.setEchoMode(
                QLineEdit.Normal if shown else QLineEdit.Password
            )
        )
        field.addAction(toggle, QLineEdit.TrailingPosition)

    def values(self) -> Dict[str, str]:
        """The entered value of each setting."""
        return {name: field.text() for name, field in self.fields.items()}

    # URL settings that receive an API key and ticket text.
    API_URL_SETTINGS = ("CHEAPER_INFERENCE_API_URL", "EMBEDDING_API_URL")

    def _api_urls_ok(self) -> bool:
        """Refuses an unencrypted API URL; asks before allowing http:// to localhost."""
        for name in self.API_URL_SETTINGS:
            url = self.fields[name].text().strip()
            if not url:
                continue
            problem = api_url_problem(url)
            if problem:
                QMessageBox.critical(self, "Insecure API URL", problem)
                return False
            # urlsplit lower-cases the scheme, so HTTP:// gets the warning too.
            if urlsplit(url).scheme == "http" and is_loopback_url(url):
                answer = QMessageBox.warning(
                    self,
                    "Unencrypted API URL",
                    f"{url} is plain http. The API key and ticket text will be sent unencrypted to a "
                    "service on this machine.\n\nUse it anyway?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    return False
        return True

    def _dimensions_ok(self) -> bool:
        """Refuses an embedding size Voyage doesn't offer."""
        value = self.fields["EMBEDDING_DIMENSIONS"].text().strip()
        if value and parse_dimensions(value) is None:
            QMessageBox.critical(
                self,
                "Invalid Dimensions",
                f"Dimensions must be one of {', '.join(map(str, ALLOWED_DIMENSIONS))}, or blank for 1024.",
            )
            return False
        return True

    def _storage_endpoint_ok(self) -> bool:
        """Refuses an unencrypted storage endpoint: the secret key signs requests to it."""
        url = self.fields["B2_ENDPOINT_URL"].text().strip()
        if (
            not url
            or url.startswith("https://")
            or (url.startswith("http://") and is_loopback_url(url))
        ):
            return True
        QMessageBox.critical(
            self,
            "Insecure Storage Endpoint",
            f"{url} must start with https:// (http:// is allowed only for localhost).",
        )
        return False

    def save(self) -> None:
        """Saves the entered values and closes, or shows why saving failed."""

        if not (
            self._api_urls_ok()
            and self._dimensions_ok()
            and self._storage_endpoint_ok()
        ):
            return
        try:
            restart_needed = settings.save(self.values())
        except OSError as e:
            QMessageBox.critical(
                self,
                "Settings Not Saved",
                f"Could not write {settings.settings_path()}:\n{e}",
            )
            return
        logger.info(f"Settings saved to {settings.settings_path()}.")
        if restart_needed:
            QMessageBox.information(
                self,
                "Restart Needed",
                "These changes take effect after you restart the app:\n"
                + "\n".join(restart_needed),
            )
        self.accept()
