from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_project_directories_exist():
    expected_dirs = [
        ROOT / "src" / "call_asr",
        ROOT / "tests" / "fixtures",
        ROOT / "data" / "submissions",
        ROOT / "data" / "runs",
        ROOT / "data" / "proxy",
    ]

    missing = [str(path) for path in expected_dirs if not path.is_dir()]

    assert missing == []


def test_package_exports_version():
    import call_asr

    assert call_asr.__version__ == "0.1.0"
