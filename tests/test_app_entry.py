"""app.py's entry point, read as source: importing app.py loads the spaCy model
and needs a display, and a spawned process importing it is the failure here."""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parents[1] / "src" / "app.py"


def _module():
    return ast.parse(APP.read_text(encoding="utf-8"))


def _is_call(node, name):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == name
    )


def test_freeze_support_runs_before_anything_else():
    # In the packaged build a multiprocessing helper (the resource tracker that
    # BERTopic's progress bars start) is launched as Lasso; freeze_support()
    # must catch it before any other import or a second app opens.
    body = _module().body
    first = next(
        i
        for i, node in enumerate(body)
        if _is_call(node, "multiprocessing.freeze_support")
    )
    before = body[:first]
    assert all(isinstance(node, ast.Import) for node in before)
    assert [alias.name for node in before for alias in node.names] == [
        "multiprocessing"
    ]


def test_no_window_is_made_on_import():
    # A spawned worker re-imports app.py as __mp_main__: only the
    # __main__ guard may build the QApplication and the window.
    module = _module()
    top_level = [
        node
        for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If))
    ]
    calls = {
        ast.unparse(call.func)
        for node in top_level
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }
    assert not calls & {"QApplication", "MainWindow", "main"}
    guard = [node for node in module.body if isinstance(node, ast.If)]
    assert [ast.unparse(node.test) for node in guard] == ["__name__ == '__main__'"]
