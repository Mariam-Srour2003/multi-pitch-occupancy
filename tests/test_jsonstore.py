"""Tests for `pitch_occupancy.jsonstore`: tolerant reads, atomic writes, history copies."""

from __future__ import annotations

import json
import os

import pytest

from pitch_occupancy import jsonstore
from pitch_occupancy.jsonstore import JSONFileError, read_json, write_json_atomic


def test_read_missing_returns_default(tmp_path):
    assert read_json(tmp_path / "nope.json", {"a": 1}) == {"a": 1}
    assert read_json(tmp_path / "nope.json") is None


def test_read_valid(tmp_path):
    target = tmp_path / "x.json"
    target.write_text('{"_comment": ["hi"], "k": [1, 2]}', encoding="utf-8")
    assert read_json(target, {}) == {"_comment": ["hi"], "k": [1, 2]}


def test_read_malformed_names_file_and_line(tmp_path):
    target = tmp_path / "broken.json"
    target.write_text('{\n  "a": 1,\n}', encoding="utf-8")
    with pytest.raises(JSONFileError) as info:
        read_json(target, {})
    assert isinstance(info.value, ValueError)
    assert "broken.json" in str(info.value)
    assert info.value.lineno == 3
    assert info.value.path == target
    assert "Expecting" in info.value.reason


def test_read_not_utf8_is_a_value_error(tmp_path):
    target = tmp_path / "bin.json"
    target.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(JSONFileError) as info:
        read_json(target)
    assert info.value.lineno is None
    assert "bin.json" in str(info.value)


def test_write_default_format(tmp_path):
    target = tmp_path / "sub" / "out.json"
    assert write_json_atomic(target, {"b": 1, "a": [1]}) is None
    assert target.read_text(encoding="utf-8") == json.dumps({"b": 1, "a": [1]}, indent=2) + "\n"
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_write_compact_options(tmp_path):
    target = tmp_path / "out.json"
    write_json_atomic(target, {"when": object}, indent=1, trailing_newline=False, default=str)
    text = target.read_text(encoding="utf-8")
    assert text == json.dumps({"when": object}, indent=1, default=str)


def test_unserialisable_leaves_old_file(tmp_path):
    target = tmp_path / "out.json"
    write_json_atomic(target, {"v": 1})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"v": object()}, history_dir=tmp_path / "hist")
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert not (tmp_path / "hist").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_history_copies_previous_and_prunes(tmp_path):
    target = tmp_path / "cams.json"
    hist = tmp_path / "cams_history"
    assert write_json_atomic(target, {"v": 0}, history_dir=hist, keep=3) is None
    assert not hist.exists()
    backups = [write_json_atomic(target, {"v": i}, history_dir=hist, keep=3)
               for i in range(1, 6)]
    kept = sorted(hist.glob("cams.*.json"))
    assert len(kept) == 3
    assert kept == backups[-3:]
    assert [json.loads(p.read_text(encoding="utf-8"))["v"] for p in kept] == [2, 3, 4]
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 5}


def test_replace_retried_on_permission_error(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held open")
        real(src, dst)

    monkeypatch.setattr(jsonstore.os, "replace", flaky)
    monkeypatch.setattr(jsonstore, "_RETRY_SLEEP_S", 0)
    write_json_atomic(target, [1], attempts=5)
    assert len(calls) == 3
    assert json.loads(target.read_text(encoding="utf-8")) == [1]


def test_replace_gives_up_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "out.json"

    def refuse(src, dst):
        raise PermissionError("held open")

    monkeypatch.setattr(jsonstore.os, "replace", refuse)
    monkeypatch.setattr(jsonstore, "_RETRY_SLEEP_S", 0)
    with pytest.raises(PermissionError):
        write_json_atomic(target, [1], attempts=2)
    assert list(tmp_path.iterdir()) == []
