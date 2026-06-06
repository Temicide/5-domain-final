from pathlib import Path

from call_asr.paths import CompetitionPaths, resolve_competition_paths


def test_resolve_local_paths_with_fixture_layout(tmp_path):
    local_root = tmp_path / "Call-ASR"
    competition_dir = local_root / "data" / "individual-test-thai-call-center-asr"
    audio_dir = competition_dir / "audio_final" / "audio"
    audio_dir.mkdir(parents=True)
    sample_submission = competition_dir / "sample_submission.csv"
    sample_submission.write_text("file_name,text\nRSP_001_audio.wav,\n", encoding="utf-8")

    paths = resolve_competition_paths(colab_input_root=tmp_path / "missing", project_root=local_root)

    assert paths == CompetitionPaths(
        input_dir=competition_dir,
        audio_dir=audio_dir,
        sample_submission=sample_submission,
        working_dir=local_root / "data" / "runs",
        submissions_dir=local_root / "data" / "submissions",
        is_colab=False,
    )


def test_resolve_colab_paths_before_local_paths(tmp_path):
    colab_input_root = tmp_path / "content" / "input"
    competition_dir = colab_input_root / "individual-test-thai-call-center-asr"
    audio_dir = competition_dir / "audio_final" / "audio"
    audio_dir.mkdir(parents=True)
    sample_submission = competition_dir / "sample_submission.csv"
    sample_submission.write_text("file_name,text\nRSP_001_audio.wav,\n", encoding="utf-8")
    local_root = tmp_path / "Call-ASR"
    (local_root / "data" / "individual-test-thai-call-center-asr" / "audio_final" / "audio").mkdir(parents=True)
    (local_root / "data" / "individual-test-thai-call-center-asr" / "sample_submission.csv").write_text(
        "file_name,text\nSDB_001_audio.wav,\n", encoding="utf-8"
    )

    paths = resolve_competition_paths(colab_input_root=colab_input_root, project_root=local_root)

    assert paths.input_dir == competition_dir
    assert paths.audio_dir == audio_dir
    assert paths.sample_submission == sample_submission
    assert paths.working_dir == Path("/content/working")
    assert paths.submissions_dir == Path("/content/working")
    assert paths.is_colab is True
