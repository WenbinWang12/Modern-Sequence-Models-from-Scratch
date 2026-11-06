"""源码与普通安装包都应能记录代码版本。"""

import pytest

from ttt.provenance import source_digest


def test_installed_package_digest_without_project_files(tmp_path):
    package = tmp_path / "ttt"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text('"""测试包。"""\n', encoding="utf-8")
    original = source_digest(tmp_path)
    assert len(original) == 64
    assert source_digest(tmp_path) == original
    (tmp_path / "unrelated.log").write_text("日志变化不影响代码版本", encoding="utf-8")
    assert source_digest(tmp_path) == original
    module.write_text('"""代码已更新。"""\n', encoding="utf-8")
    assert source_digest(tmp_path) != original


def test_source_digest_tracks_dependency_lock(tmp_path):
    (tmp_path / "ttt").mkdir()
    (tmp_path / "ttt" / "__init__.py").write_text("", encoding="utf-8")
    original = source_digest(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    locked = source_digest(tmp_path)
    assert locked != original
    lock.write_text("version = 2\n", encoding="utf-8")
    assert source_digest(tmp_path) != locked


def test_source_digest_rejects_missing_package(tmp_path):
    with pytest.raises(ValueError, match="ttt"):
        source_digest(tmp_path)
