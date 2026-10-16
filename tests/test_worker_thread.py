import os
import subprocess
import sys
import textwrap

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import utility


def test_worker_threads_get_a_large_stack():
    thread = utility.worker_thread()
    assert thread.stackSize() == utility.WORKER_STACK_BYTES >= 8 * 1024 * 1024


def test_numerical_code_runs_on_a_worker_thread():
    """OpenBLAS's LU (numpy.linalg.inv, as matplotlib uses) overflowed the 512 KiB
    default QThread stack on macOS and killed the app with SIGBUS. Run in a child
    process so a crash fails the test instead of the test run."""
    script = textwrap.dedent("""
        import numpy as np
        from PyQt5.QtCore import QCoreApplication
        import utility

        app = QCoreApplication([])
        results = []
        # As app.py runs jobs: a ThreadWorker moved onto a worker thread.
        thread = utility.worker_thread()
        worker = utility.ThreadWorker(
            lambda: [np.linalg.inv(np.random.rand(n, n) + n * np.eye(n)).shape for n in (3, 200, 400)], "inverting")
        worker.moveToThread(thread)
        worker.finished.connect(lambda shapes: (results.append(shapes), app.quit()))
        worker.error.connect(lambda message: (results.append(message), app.quit()))
        thread.started.connect(worker.run)
        thread.start()
        app.exec_()
        thread.quit()
        thread.wait()
        assert results == [[(3, 3), (200, 200), (400, 400)]], results
    """)
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    result = subprocess.run([sys.executable, "-c", script], cwd=src, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": src})
    assert result.returncode == 0, result.stderr[-2000:]
