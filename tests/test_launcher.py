import importlib.util
from pathlib import Path


def test_launcher_resolves_root_main():
    path = Path(__file__).parents[1] / "launcher.py"
    spec = importlib.util.spec_from_file_location("launcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.PROJECT_ROOT / "main.py" == Path(__file__).parents[1] / "main.py"
    assert module.project_python() == Path(__file__).parents[1] / ".venv" / "bin" / "python"
