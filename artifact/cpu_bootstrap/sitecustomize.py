"""Private Ray-worker bootstrap, enabled only by the CPU scheduling entry."""
import os

if os.environ.get('MIL_CPU_SCHEDULING') == '1':
    from artifact.mil_artifact.cpu_guard import install
    install()
