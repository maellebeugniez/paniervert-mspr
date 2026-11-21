from pathlib import Path


def test_j3_files_exist():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "dvc.yaml",
        root / "src" / "api" / "service.py",
        root / "src" / "monitoring" / "drift_report.py",
        root / "src" / "quality_gate.py",
        root / ".github" / "workflows" / "ci.yml",
        root / "docker-compose.yml",
    ]
    missing = [str(path.relative_to(root)) for path in paths if not path.exists()]
    assert not missing, f"Fichiers J3 manquants : {missing}"


def test_mlruns_is_ignored():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/mlruns/" in gitignore
