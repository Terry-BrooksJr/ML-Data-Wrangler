"""Settings dialog: edit the environment variables in settings.SETTINGS."""

from itertools import groupby
from typing import Dict

from loguru import logger
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import settings
from decision_engine.classify import api_url_problem, is_loopback_url


class SettingsDialog(QDialog):
    """Lists every setting by group; Save writes them via settings.save().

    Required fields are marked with *. Secret fields are masked, with an eye
    toggle to reveal them.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(620, 640)
        self.fields: Dict[str, QLineEdit] = {}

        intro = QLabel(
            "Fields marked * are needed by the Decision Engine (Mongo URI also by "
            "Write Tickets to MongoDB). Leave a field blank to use its default, even if "
            f".env or your shell sets it. Saved to:\n{settings.settings_path()}"
        )
        intro.setWordWrap(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        values = settings.current()
        for group, items in groupby(settings.SETTINGS, key=lambda s: s.group):
            box = QGroupBox(group)
            form = QFormLayout(box)
            for setting in items:
                field = QLineEdit(values[setting.name])
                field.setPlaceholderText(setting.default)
                field.setClearButtonEnabled(True)
                tooltip = setting.name + (f"\n{setting.help}" if setting.help else "")
                if setting.restart:
                    tooltip += "\nChanges take effect after a restart."
                field.setToolTip(tooltip)
                if setting.secret:
                    self._make_secret(field)
                label = setting.label + (" *" if setting.required else "")
                form.addRow(label, field)
                self.fields[setting.name] = field
            content_layout.addWidget(box)
        content_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
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
        toggle.setIcon(field.style().standardIcon(field.style().SP_FileDialogContentsView))
        toggle.toggled.connect(
            lambda shown: field.setEchoMode(QLineEdit.Normal if shown else QLineEdit.Password)
        )
        field.addAction(toggle, QLineEdit.TrailingPosition)

    def values(self) -> Dict[str, str]:
        """The entered value of each setting."""
        return {name: field.text() for name, field in self.fields.items()}

    def _api_url_ok(self) -> bool:
        """Refuses an unencrypted API URL; asks before allowing http:// to localhost."""
        url = self.fields["CHEAPER_INFERENCE_API_URL"].text().strip()
        if not url:
            return True
        problem = api_url_problem(url)
        if problem:
            QMessageBox.critical(self, "Insecure API URL", problem)
            return False
        if url.startswith("http://") and is_loopback_url(url):
            answer = QMessageBox.warning(
                self,
                "Unencrypted API URL",
                f"{url} is plain http. The API key and ticket text will be sent unencrypted to a "
                "service on this machine.\n\nUse it anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            return answer == QMessageBox.Yes
        return True

    def save(self) -> None:
        """Saves the entered values and closes, or shows why saving failed."""
        if not self._api_url_ok():
            return
        try:
            restart_needed = settings.save(self.values())
        except OSError as e:
            QMessageBox.critical(self, "Settings Not Saved", f"Could not write {settings.settings_path()}:\n{e}")
            return
        logger.info(f"Settings saved to {settings.settings_path()}.")
        if restart_needed:
            QMessageBox.information(
                self,
                "Restart Needed",
                "These changes take effect after you restart the app:\n" + "\n".join(restart_needed),
            )
        self.accept()
