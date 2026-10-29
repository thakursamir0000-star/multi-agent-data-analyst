"""
Child process for tools.sandbox.run_code (process isolation mode).

Reads one pickled {code, local_vars, timeout, max_memory_mb} payload from
stdin — written by the trusted parent after its AST security scan — applies
resource limits, runs the snippet with the same restricted namespace as the
in-process mode, prints the result as JSON after a marker line, and exits.
The parent only ever parses JSON from this process, never pickle.
"""

import json
import os
import pickle
import sys
import threading
from pathlib import Path


def _limit_resources(max_memory_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # Windows: the watchdog and the parent's timeout still apply
        return
    memory = max_memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))


# The real stdout, saved before the snippet runs with sys.stdout redirected to a
# capture buffer (the watchdog fires while that redirect is active). Raw bytes,
# because text-mode stdout on Windows would turn the marker's "\n" into "\r\n".
_REAL_STDOUT = sys.stdout.buffer


def _emit(marker: str, result: dict) -> None:
    _REAL_STDOUT.write((marker + json.dumps(result)).encode("utf-8"))
    _REAL_STDOUT.flush()


def main() -> None:
    payload = pickle.loads(sys.stdin.buffer.read())
    timeout = int(payload["timeout"])
    _limit_resources(int(payload["max_memory_mb"]), timeout + 5)

    # -I keeps the project off sys.path; add it back explicitly for tools.sandbox.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.sandbox import RESULT_MARKER, _timeout_result, execute_snippet

    def on_timeout() -> None:
        _emit(RESULT_MARKER, _timeout_result(timeout))
        os._exit(0)

    # The clock starts after interpreter start-up, so slow imports don't eat the budget.
    watchdog = threading.Timer(timeout, on_timeout)
    watchdog.daemon = True
    watchdog.start()
    result = execute_snippet(payload["code"], payload["local_vars"])
    watchdog.cancel()
    _emit(RESULT_MARKER, result)


if __name__ == "__main__":
    main()
