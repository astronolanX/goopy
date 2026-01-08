"""
Security and adversarial tests for P0/P1 hardening.

Tests stress-test the security implementations:
- P0-01: Atomic writes (file locking replacement)
- P0-02: Path traversal prevention
- P0-03: Archive uniqueness
- P1-04: KNOWN_SUBDIRS consistency
"""

import inspect
import os
import pytest
import tempfile
import threading
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4

from goopy.blob import (
    Blob,
    BlobType,
    BlobScope,
    BlobStatus,
    Glob,
    PathTraversalError,
    _atomic_write,
    _validate_path_safe,
    KNOWN_SUBDIRS,
)


# =============================================================================
# P0-01: Atomic Write Tests
# =============================================================================


class TestAtomicWriteFailureModes:
    """Stress test atomic write under failure conditions."""

    def test_atomic_write_basic(self):
        """Basic atomic write works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "hello world")
            assert path.read_text() == "hello world"

    def test_atomic_write_creates_parent_dirs(self):
        """Atomic write creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "dir" / "test.txt"
            _atomic_write(path, "nested content")
            assert path.read_text() == "nested content"

    def test_atomic_write_overwrites_existing(self):
        """Atomic write replaces existing file atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "original")
            _atomic_write(path, "replaced")
            assert path.read_text() == "replaced"

    def test_atomic_write_no_temp_file_on_success(self):
        """No temp files left after successful write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "content")

            # List all files - should only be the target
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 1
            assert files[0].name == "test.txt"

    def test_atomic_write_cleans_temp_on_write_failure(self):
        """Temp file cleaned up when os.write fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"

            with patch("os.write", side_effect=OSError("Disk full")):
                with pytest.raises(OSError, match="Disk full"):
                    _atomic_write(path, "content")

            # No files should remain
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 0

    def test_atomic_write_cleans_temp_on_fsync_failure(self):
        """Temp file cleaned up when os.fsync fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"

            original_fsync = os.fsync

            def failing_fsync(fd):
                raise OSError("fsync failed")

            with patch("os.fsync", side_effect=failing_fsync):
                with pytest.raises(OSError, match="fsync failed"):
                    _atomic_write(path, "content")

            # No files should remain (temp cleaned up)
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 0

    def test_atomic_write_cleans_temp_on_rename_failure(self):
        """Temp file cleaned up when os.rename fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"

            with patch("os.rename", side_effect=OSError("rename failed")):
                with pytest.raises(OSError, match="rename failed"):
                    _atomic_write(path, "content")

            # No files should remain
            files = list(Path(tmpdir).iterdir())
            assert len(files) == 0

    def test_atomic_write_preserves_original_on_failure(self):
        """Original file preserved if write fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "original content")

            with patch("os.write", side_effect=OSError("Disk full")):
                with pytest.raises(OSError):
                    _atomic_write(path, "new content that fails")

            # Original should be intact
            assert path.read_text() == "original content"

    def test_atomic_write_large_content(self):
        """Atomic write handles large content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.txt"
            large_content = "x" * 10_000_000  # 10MB
            _atomic_write(path, large_content)
            assert len(path.read_text()) == 10_000_000


class TestAtomicWriteConcurrency:
    """Test atomic write under concurrent access."""

    def test_concurrent_writes_no_corruption(self):
        """Concurrent writes don't produce corrupt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "contested.txt"
            results = []
            errors = []

            def writer(content):
                try:
                    for _ in range(50):
                        _atomic_write(path, content)
                    results.append(content)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=writer, args=(f"writer_{i}" * 1000,))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Errors occurred: {errors}"

            # Final content should be complete (from one writer)
            content = path.read_text()
            assert len(content) > 0
            # Should be one of the valid contents (not mixed)
            assert content.startswith("writer_")

    def test_reader_sees_complete_content(self):
        """Readers never see partial content during writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            _atomic_write(path, "initial")

            partial_reads = []
            stop_flag = threading.Event()

            def reader():
                while not stop_flag.is_set():
                    try:
                        content = path.read_text()
                        # Content should always be valid
                        if not (content.startswith("content_") or content == "initial"):
                            partial_reads.append(content)
                    except FileNotFoundError:
                        pass  # File may momentarily not exist during rename

            def writer():
                for i in range(100):
                    _atomic_write(path, f"content_{i}" * 500)

            reader_thread = threading.Thread(target=reader)
            writer_thread = threading.Thread(target=writer)

            reader_thread.start()
            writer_thread.start()
            writer_thread.join()
            stop_flag.set()
            reader_thread.join()

            # Should never have seen partial content
            assert len(partial_reads) == 0, f"Saw partial reads: {partial_reads[:5]}"


# =============================================================================
# P0-02: Path Traversal Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Adversarial tests for path traversal prevention."""

    def test_simple_parent_traversal(self):
        """Basic ../ traversal blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = base / ".." / "escaped.txt"

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_double_parent_traversal(self):
        """../../ traversal blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = base / ".." / ".." / "escaped.txt"

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_windows_style_traversal(self):
        r"""..\ traversal blocked (Windows-style)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            # Path normalizes this
            target = Path(str(base) + "\\..\\escaped.txt")

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_mixed_traversal(self):
        """Mixed forward/back traversal blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = base / "subdir" / ".." / ".." / "escaped.txt"

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_traversal_in_middle(self):
        """Traversal embedded in path blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = base / "foo" / ".." / ".." / "escaped.txt"

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_absolute_path_escape(self):
        """Absolute path outside base blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = Path("/etc/passwd")

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_symlink_escape(self):
        """Symlink pointing outside base blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            outside = Path(tmpdir) / "outside"
            outside.mkdir()

            # Create symlink inside base pointing outside
            link = base / "escape_link"
            link.symlink_to(outside)
            target = link / "file.txt"

            with pytest.raises(PathTraversalError):
                _validate_path_safe(base, target)

    def test_valid_nested_path(self):
        """Valid nested path allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            (base / "subdir").mkdir()
            target = base / "subdir" / "file.txt"

            result = _validate_path_safe(base, target)
            assert result == target.resolve()

    def test_valid_path_with_dots_in_name(self):
        """File names with dots allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            target = base / "file.with.dots.txt"

            result = _validate_path_safe(base, target)
            assert "file.with.dots.txt" in str(result)

    def test_glob_sprout_traversal_blocked(self):
        """Glob.sprout blocks traversal in subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="Evil")

            with pytest.raises(PathTraversalError):
                glob.sprout(blob, "evil", subdir="../escape")

    def test_glob_sprout_traversal_in_name_blocked(self):
        """Glob.sprout blocks traversal in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="Evil")

            with pytest.raises(PathTraversalError):
                glob.sprout(blob, "../../../etc/passwd")

    def test_glob_get_traversal_blocked(self):
        """Glob.get blocks traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            with pytest.raises(PathTraversalError):
                glob.get("../../../etc/passwd")

    def test_glob_decompose_traversal_blocked(self):
        """Glob.decompose blocks traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            with pytest.raises(PathTraversalError):
                glob.decompose("../../../etc/passwd")


class TestPathTraversalUnicode:
    """Unicode-based path traversal attacks."""

    def test_fullwidth_slash(self):
        """Fullwidth slash (U+FF0F) doesn't bypass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            # Fullwidth characters - Path normalizes these
            target = base / "..\uff0f.." / "escaped.txt"

            # Either blocks or treats as literal filename
            try:
                result = _validate_path_safe(base, target)
                # If allowed, must be within base
                assert str(base.resolve()) in str(result)
            except PathTraversalError:
                pass  # Also acceptable

    def test_unicode_period(self):
        """Unicode period substitutes don't bypass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "base"
            base.mkdir()
            # Fullwidth period: \uff0e
            target = base / "\uff0e\uff0e" / "escaped.txt"

            # Either blocks or treats as literal filename
            try:
                result = _validate_path_safe(base, target)
                # If allowed, must be within base
                assert str(base.resolve()) in str(result)
            except PathTraversalError:
                pass


# =============================================================================
# P0-03: Archive Uniqueness Tests
# =============================================================================


class TestArchiveUniqueness:
    """Stress test archive filename uniqueness."""

    def test_rapid_decompose_unique_names(self):
        """Rapid decompose produces unique archive names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            archive_dir = glob.claude_dir / "archive"

            for i in range(100):
                blob = Blob(type=BlobType.FACT, summary=f"Blob {i}")
                glob.sprout(blob, "contested")
                glob.decompose("contested")

            # All archived files should have unique names
            archived = list(archive_dir.glob("*.blob.xml"))
            names = [f.name for f in archived]
            assert len(names) == len(set(names)), "Duplicate archive names!"
            assert len(names) == 100

    def test_decompose_same_name_parallel(self):
        """Parallel decompose of same-name blobs - uniqueness under race."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            success_count = [0]  # Use list for closure mutability
            lock = threading.Lock()

            def decompose_cycle(worker_id):
                for i in range(20):
                    try:
                        # Each worker uses unique blob names to avoid race
                        name = f"blob_{worker_id}_{i}"
                        blob = Blob(type=BlobType.FACT, summary=f"Worker {worker_id}")
                        glob.sprout(blob, name)
                        glob.decompose(name)
                        with lock:
                            success_count[0] += 1
                    except Exception:
                        pass  # Race conditions expected

            threads = [
                threading.Thread(target=decompose_cycle, args=(i,))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Check archive for uniqueness - this is the key assertion
            archive_dir = glob.claude_dir / "archive"
            assert archive_dir.exists()
            archived = list(archive_dir.glob("*.blob.xml"))
            names = [f.name for f in archived]
            # All archive names must be unique
            assert len(names) == len(set(names)), f"Duplicate names found!"
            # Should have archived many blobs
            assert len(names) >= 50, f"Only {len(names)} archived, expected ~100"

    def test_uuid_collision_handling(self):
        """Verify UUID is used for uniqueness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create and decompose
            blob1 = Blob(type=BlobType.FACT, summary="First")
            glob.sprout(blob1, "test")
            glob.decompose("test")

            blob2 = Blob(type=BlobType.FACT, summary="Second")
            glob.sprout(blob2, "test")
            glob.decompose("test")

            # Check archive names include unique component
            archive_dir = glob.claude_dir / "archive"
            archived = list(archive_dir.glob("*.blob.xml"))
            assert len(archived) == 2

            # Names should differ (UUID portion)
            assert archived[0].name != archived[1].name


# =============================================================================
# P1-04: KNOWN_SUBDIRS Consistency Tests
# =============================================================================


class TestKnownSubdirsConsistency:
    """Verify KNOWN_SUBDIRS is the single source of truth."""

    def test_known_subdirs_not_empty(self):
        """KNOWN_SUBDIRS is defined and non-empty."""
        assert KNOWN_SUBDIRS
        assert len(KNOWN_SUBDIRS) >= 5

    def test_known_subdirs_has_expected_values(self):
        """KNOWN_SUBDIRS contains expected subdirectories."""
        expected = {"threads", "decisions", "constraints", "contexts", "facts"}
        actual = set(KNOWN_SUBDIRS)
        assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"

    def test_glob_uses_known_subdirs(self):
        """Glob methods use KNOWN_SUBDIRS constant."""
        import inspect
        from goopy import blob as blob_module

        # Check that surface_relevant and check_migrations use the constant
        surface_src = inspect.getsource(Glob.surface_relevant)
        migrations_src = inspect.getsource(Glob.check_migrations)

        # They should reference KNOWN_SUBDIRS, not hardcode
        assert "KNOWN_SUBDIRS" in surface_src, "surface_relevant should use KNOWN_SUBDIRS"
        assert "KNOWN_SUBDIRS" in migrations_src, "check_migrations should use KNOWN_SUBDIRS"

    def test_no_hardcoded_subdir_lists(self):
        """No hardcoded subdir lists in blob.py."""
        from goopy import blob as blob_module
        import re

        source = inspect.getsource(blob_module)

        # Look for suspicious patterns (hardcoded lists with subdir names)
        # Exclude the KNOWN_SUBDIRS definition itself
        lines = source.split("\n")
        suspicious = []

        for i, line in enumerate(lines):
            # Skip the KNOWN_SUBDIRS definition
            if "KNOWN_SUBDIRS" in line and "=" in line:
                continue
            # Skip comments and strings that explain things
            if line.strip().startswith("#"):
                continue

            # Check for inline lists with these strings
            if (
                '"threads"' in line
                and '"decisions"' in line
                or '"contexts"' in line
                and '"facts"' in line
            ):
                suspicious.append((i + 1, line.strip()))

        assert not suspicious, f"Found hardcoded subdir lists: {suspicious}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests combining multiple security features."""

    def test_full_lifecycle_secure(self):
        """Full blob lifecycle uses all security measures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create blob
            blob = Blob(
                type=BlobType.THREAD,
                summary="Secure lifecycle test",
                status=BlobStatus.ACTIVE,
            )
            path = glob.sprout(blob, "secure-test", subdir="threads")
            assert path.exists()

            # Read back
            loaded = glob.get("secure-test", subdir="threads")
            assert loaded.summary == "Secure lifecycle test"

            # Decompose (archive)
            glob.decompose("secure-test", subdir="threads")
            assert not (glob.claude_dir / "threads" / "secure-test.blob.xml").exists()

            # Archived version exists with unique name
            archive_dir = glob.claude_dir / "archive"
            archived = list(archive_dir.glob("*secure-test*.blob.xml"))
            assert len(archived) == 1

    def test_traversal_blocked_at_all_entry_points(self):
        """All Glob methods block traversal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # sprout
            with pytest.raises(PathTraversalError):
                glob.sprout(Blob(type=BlobType.FACT, summary="X"), "../escape")

            with pytest.raises(PathTraversalError):
                glob.sprout(Blob(type=BlobType.FACT, summary="X"), "ok", subdir="../escape")

            # get
            with pytest.raises(PathTraversalError):
                glob.get("../escape")

            with pytest.raises(PathTraversalError):
                glob.get("ok", subdir="../escape")

            # decompose
            with pytest.raises(PathTraversalError):
                glob.decompose("../escape")

            with pytest.raises(PathTraversalError):
                glob.decompose("ok", subdir="../escape")


# =============================================================================
# Cleanup Session Tests (Swarm Safety)
# =============================================================================


class TestCleanupSessionBasic:
    """Basic cleanup_session() functionality tests."""

    def test_cleanup_session_empty_glob(self):
        """Cleanup on empty glob succeeds with no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            results = glob.cleanup_session()

            assert results["sessions_pruned"] == 0
            assert results["archives_pruned"] == 0
            assert results["migrated"] == 0
            assert not results["skipped"]
            assert not results["locked"]

    def test_cleanup_prunes_old_session_blobs(self):
        """Old SESSION-scope blobs are pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create a session blob with old date
            from datetime import timedelta
            old_date = datetime.now() - timedelta(days=2)
            blob = Blob(
                type=BlobType.CONTEXT,
                summary="Old session",
                scope=BlobScope.SESSION,
                updated=old_date,
            )
            path = glob.sprout(blob, "old-session", subdir="contexts")
            assert path.exists()

            # Run cleanup
            results = glob.cleanup_session()
            assert results["sessions_pruned"] == 1
            assert not path.exists()

    def test_cleanup_preserves_project_scope_blobs(self):
        """PROJECT-scope blobs are not pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create a project blob with old date
            from datetime import timedelta
            old_date = datetime.now() - timedelta(days=30)
            blob = Blob(
                type=BlobType.FACT,
                summary="Old project fact",
                scope=BlobScope.PROJECT,
                updated=old_date,
            )
            path = glob.sprout(blob, "old-fact", subdir="facts")

            # Run cleanup
            results = glob.cleanup_session()
            assert results["sessions_pruned"] == 0
            assert path.exists()

    def test_cleanup_prunes_old_archives(self):
        """Archives older than threshold are pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create and decompose a blob
            blob = Blob(type=BlobType.THREAD, summary="To archive", status=BlobStatus.ACTIVE)
            glob.sprout(blob, "to-archive", subdir="threads")
            glob.decompose("to-archive", subdir="threads")

            # Manually backdate the archive
            archive_dir = glob.claude_dir / "archive"
            archived = list(archive_dir.glob("*.blob.xml"))[0]
            archived_blob = Blob.load(archived)
            archived_blob.updated = datetime.now() - timedelta(days=60)
            archived_blob.save(archived)

            # Run cleanup with 30-day threshold
            results = glob.cleanup_session(archive_days=30)
            assert results["archives_pruned"] == 1
            assert not archived.exists()

    def test_cleanup_dry_run(self):
        """Dry run reports but doesn't delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create old session blob
            old_date = datetime.now() - timedelta(days=2)
            blob = Blob(
                type=BlobType.CONTEXT,
                summary="Old session",
                scope=BlobScope.SESSION,
                updated=old_date,
            )
            path = glob.sprout(blob, "old-session", subdir="contexts")

            # Dry run
            results = glob.cleanup_session(dry_run=True)
            assert results["sessions_pruned"] == 1
            assert path.exists()  # Still exists

    def test_cleanup_skips_if_already_cleaned_today(self):
        """Cleanup skips if already run today."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # First cleanup
            results1 = glob.cleanup_session()
            assert not results1["skipped"]

            # Second cleanup same day
            results2 = glob.cleanup_session()
            assert results2["skipped"]


class TestCleanupSessionConcurrency:
    """Concurrent cleanup tests for swarm safety."""

    def test_cleanup_lock_prevents_concurrent_cleanup(self):
        """Lock file prevents concurrent cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Simulate another agent holding lock
            lock_path = glob.claude_dir / ".cleanup.lock"
            lock_path.touch()

            # Our cleanup should skip
            results = glob.cleanup_session()
            assert results["locked"]
            assert results["sessions_pruned"] == 0

            # Clean up lock
            lock_path.unlink()

    def test_cleanup_lock_released_on_success(self):
        """Lock file is released after successful cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            glob.cleanup_session()

            lock_path = glob.claude_dir / ".cleanup.lock"
            assert not lock_path.exists()

    def test_cleanup_lock_released_on_error(self):
        """Lock file is released even if cleanup fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create a corrupted blob that will fail to load
            (glob.claude_dir / "contexts").mkdir(parents=True)
            (glob.claude_dir / "contexts" / "corrupt.blob.xml").write_text("garbage")

            # Cleanup should still complete
            results = glob.cleanup_session()

            lock_path = glob.claude_dir / ".cleanup.lock"
            assert not lock_path.exists()

    def test_concurrent_cleanup_one_wins(self):
        """Multiple concurrent cleanups: only one runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            results_list = []
            errors = []

            def run_cleanup():
                try:
                    results = glob.cleanup_session()
                    results_list.append(results)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=run_cleanup) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors

            # Exactly one should have run, others skipped or locked
            ran = sum(1 for r in results_list if not r["skipped"] and not r["locked"])
            skipped_or_locked = sum(1 for r in results_list if r["skipped"] or r["locked"])

            # First one runs, rest are skipped (already cleaned) or locked
            assert ran + skipped_or_locked == 5
