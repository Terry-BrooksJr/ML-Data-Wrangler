"""The PyQt-SiliconUI widgets the app uses, with the few gaps it needs filled.

Each class subclasses a SiliconUI widget, which subclasses the Qt widget it
stands in for (QPushButton, QComboBox, QAbstractButton, QProgressBar,
QLineEdit), so callers keep using the Qt API: clicked, setEnabled,
currentData, setChecked, toggled, text, returnPressed.
"""

from typing import Optional

import theme  # noqa: F401  (imports siui quietly; see theme.py)
from PyQt5.QtCore import QEvent, QObject, QSize
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QBoxLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from siui.components.button import SiCheckBox, SiPushButtonRefactor
from siui.components.combobox_ import SiCapsuleComboBox
from siui.components.container import SiPanelCard
from siui.components.editbox import SiCapsuleLineEdit
from siui.components.progress_bar_ import SiProgressBarRefactor
from siui.components.slider_ import SiScrollBar

DISABLED_OPACITY = 0.4


def _adopt_helpers(widget: QWidget) -> None:
    """Parents the helper QObjects SiliconUI leaves parentless to their widget.

    Some helpers (tooltip event filters, the progress bar's flash timer) are
    referenced only from Python. Python could free one while Qt was still
    deleting its widget, which crashed; as children, Qt deletes them in order.
    """
    for owner in [widget, *widget.findChildren(QWidget)]:
        for value in list(vars(owner).values()):
            if (
                isinstance(value, QObject)
                and not isinstance(value, QWidget)
                and value.parent() is None
            ):
                value.setParent(owner)


def _dim_when_disabled(widget: QWidget) -> None:
    """SiliconUI paints its widgets the same enabled or not; fade disabled ones."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
    effect.setOpacity(1.0 if widget.isEnabled() else DISABLED_OPACITY)


class Button(SiPushButtonRefactor):
    """A SiliconUI push button; primary=True gives it the accent colour."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, primary: bool = False) -> None:
        super().__init__(parent)
        self.setText(text)
        if primary:
            self.style_data.button_color = QColor(theme.ACCENT)
            self.style_data.background_color = QColor("#52389a")
        _dim_when_disabled(self)
        _adopt_helpers(self)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width() + 16, max(hint.height(), 34))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.EnabledChange:
            _dim_when_disabled(self)


class CheckBox(SiCheckBox):
    """A SiliconUI option-style checkbox."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setText(text)
        _dim_when_disabled(self)
        _adopt_helpers(self)

    def sizeHint(self) -> QSize:
        # SiliconUI sizes the text exactly, so rounding elides its last letters.
        hint = super().sizeHint()
        return QSize(hint.width() + 8, hint.height())

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.EnabledChange:
            _dim_when_disabled(self)


class ComboBox(SiCapsuleComboBox):
    """A SiliconUI capsule combo box with its title on the left."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle(title)
        self.setMinimumHeight(32)
        _adopt_helpers(self)


class LineEdit(SiCapsuleLineEdit):
    """A SiliconUI capsule line edit with its title on the left."""

    def __init__(self, title: str, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, title)
        self.setText(text)
        self.setMinimumHeight(32)
        _dim_when_disabled(self)
        _adopt_helpers(self)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.EnabledChange:
            _dim_when_disabled(self)


class ProgressBar(SiProgressBarRefactor):
    """A SiliconUI progress bar that also does Qt's busy mode (range 0..0).

    SiliconUI divides by the range, so 0..0 would raise; here it fills the bar
    and pulses while shown.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(6)
        _adopt_helpers(self)

    def _valueToProgress(self, value: float) -> float:
        if self.maximum() == self.minimum():
            return 1.0
        return super()._valueToProgress(value)

    def setRange(self, minimum: int, maximum: int) -> None:
        super().setRange(minimum, maximum)
        self.ani_prog_value.setEndValue(self._valueToProgress(self.value()))
        self.ani_prog_value.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.setFlashing(self.maximum() == self.minimum())

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.setFlashing(False)


class Card(SiPanelCard):
    """A SiliconUI panel card holding a column of widgets, with an optional heading.

    Add to card.body, a QVBoxLayout.
    """

    def __init__(self, title: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, QBoxLayout.TopToBottom)
        self.stretchWidget().hide()
        self.muteStretchWidget()
        self.layout().setContentsMargins(16, 12, 16, 16)
        if title:
            self.layout().addWidget(heading(title))
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        self.layout().addLayout(self.body)


def heading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "heading")
    return label


def hint(text: str, role: str = "hint") -> QLabel:
    """A wrapped, dimmed label (role="warning" for a caution)."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", role)
    return label


def use_si_scrollbars(view: QAbstractScrollArea) -> QAbstractScrollArea:
    """Gives a table, tree or text box SiliconUI's scroll bars; returns it."""
    for set_bar in (view.setVerticalScrollBar, view.setHorizontalScrollBar):
        bar = SiScrollBar(view)
        set_bar(bar)
        _adopt_helpers(bar)
    return view


def button_row(*buttons: QWidget) -> QHBoxLayout:
    """The buttons left to right, with the space after them."""
    row = QHBoxLayout()
    for button in buttons:
        row.addWidget(button)
    row.addStretch()
    return row
