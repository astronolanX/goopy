"""
Absurd and adversarial test cases.
Testing edge cases that should never happen but might.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

from goopy.blob import Blob, BlobType, BlobScope, BlobStatus, Glob, BLOB_VERSION


class TestAbsurdInputs:
    """Tests with absurd, malformed, or adversarial inputs."""

    def test_blob_with_only_whitespace_summary(self):
        """Summary that's only whitespace."""
        blob = Blob(type=BlobType.FACT, summary="   \n\t  ")
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        # Whitespace may be normalized

    def test_blob_summary_is_xml(self):
        """Summary that looks like XML."""
        blob = Blob(type=BlobType.FACT, summary="<script>alert('xss')</script>")
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert "<script>" in restored.summary or "&lt;script&gt;" in xml

    def test_blob_context_is_malicious_xml(self):
        """Context with XML injection attempt."""
        blob = Blob(
            type=BlobType.CONTEXT,
            summary="Test",
            context='</context><hacked>true</hacked><context>'
        )
        xml = blob.to_xml()
        # Should be escaped, not interpreted
        assert "<hacked>" not in xml or "&lt;hacked&gt;" in xml
        restored = Blob.from_xml(xml)
        assert "hacked" in restored.context

    def test_billion_laughs_defense(self):
        """XML billion laughs attack (entity expansion)."""
        malicious_xml = '''<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<blob type="fact" scope="project" v="2" updated="2025-01-01">
  <summary>&lol2;</summary>
</blob>'''
        # Should either reject or handle safely
        try:
            blob = Blob.from_xml(malicious_xml)
            # If it parses, summary should be reasonable size
            assert len(blob.summary) < 10000
        except ET.ParseError:
            pass  # Expected - rejection is fine

    def test_xxe_attack_defense(self):
        """XXE (XML External Entity) attack."""
        malicious_xml = '''<?xml version="1.0"?>
<!DOCTYPE blob [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<blob type="fact" scope="project" v="2" updated="2025-01-01">
  <summary>&xxe;</summary>
</blob>'''
        # Should either reject or not include file contents
        try:
            blob = Blob.from_xml(malicious_xml)
            assert "root:" not in blob.summary  # /etc/passwd indicator
        except ET.ParseError:
            pass  # Expected - rejection is fine

    def test_file_path_traversal(self):
        """File paths with traversal attempts."""
        blob = Blob(
            type=BlobType.THREAD,
            summary="Traversal test",
            files=["../../../etc/passwd", "..\\..\\windows\\system32\\config"]
        )
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        # Paths should be stored as-is (validation is on read)
        assert "../../../etc/passwd" in restored.files

    def test_extremely_deep_nesting_summary(self):
        """Summary with deeply nested brackets."""
        deep = "".join(["(((" for _ in range(1000)]) + "center" + "".join([")))" for _ in range(1000)])
        blob = Blob(type=BlobType.FACT, summary=deep)
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert "center" in restored.summary

    def test_null_in_file_list(self):
        """Null-like values in file list."""
        # Python doesn't have null, but test empty strings
        blob = Blob(type=BlobType.THREAD, summary="Null files", files=["", "real.py", ""])
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        # Empty strings might be filtered or preserved

    def test_decision_with_newlines_in_why(self):
        """Decision with multiline 'why' in attribute."""
        blob = Blob(
            type=BlobType.DECISION,
            summary="Test",
            decisions=[("choice", "line1\nline2\nline3")]
        )
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        # Newlines in attributes may be normalized to spaces

    def test_extremely_long_attribute(self):
        """Attribute with extremely long value."""
        blob = Blob(
            type=BlobType.DECISION,
            summary="Long attr",
            decisions=[("choice", "x" * 100000)]
        )
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert len(restored.decisions[0][1]) == 100000

    def test_emoji_everywhere(self):
        """Emojis in every field."""
        blob = Blob(
            type=BlobType.THREAD,
            summary="🚀 Launch 🎉",
            context="🔥 Context with 💯 emojis 🎸",
            files=["🎵/music.py", "🎨/art.py"],
            decisions=[("🍕 Pizza", "Because 🤤")],
            facts=["🌍 Earth is round"],
            next_steps=["🏃 Run", "🚶 Walk"],
            blocked_by="🚧 Construction",
            related=["🔗-link", "⛓️-chain"],
            status=BlobStatus.ACTIVE,
        )
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert "🚀" in restored.summary
        assert "🔥" in restored.context

    def test_mixed_encodings_in_content(self):
        """Mixed encoding characters."""
        blob = Blob(
            type=BlobType.FACT,
            summary="ASCII 日本語 Ελληνικά العربية עברית"
        )
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert "日本語" in restored.summary
        assert "Ελληνικά" in restored.summary

    def test_blob_name_with_dots(self):
        """Blob name with multiple dots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="Dots")
            path = glob.sprout(blob, "my.blob.v1.0.final")
            assert path.exists()
            # Name parsing should still work
            assert "my.blob.v1.0.final" in str(path)

    def test_blob_name_is_blob(self):
        """Blob named 'blob'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="Meta")
            path = glob.sprout(blob, "blob")
            assert path.exists()
            retrieved = glob.get("blob")
            assert retrieved.summary == "Meta"

    def test_subdir_is_archive(self):
        """Using 'archive' as subdir (conflicts with decompose)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="In archive")
            path = glob.sprout(blob, "test", subdir="archive")
            assert path.exists()
            # This might cause confusion with decompose

    def test_circular_related_refs(self):
        """Blobs referencing each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            blob_a = Blob(type=BlobType.FACT, summary="A", related=["b"])
            blob_b = Blob(type=BlobType.FACT, summary="B", related=["a"])

            glob.sprout(blob_a, "a")
            glob.sprout(blob_b, "b")

            # Should not cause infinite loops
            relevant = glob.surface_relevant(query="test")

    def test_self_referencing_blob(self):
        """Blob that references itself."""
        blob = Blob(type=BlobType.FACT, summary="Self ref", related=["self-ref"])
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert "self-ref" in restored.related


class TestAbsurdGlobOperations:
    """Absurd operations on Glob."""

    def test_sprout_to_parent_directory(self):
        """Attempting to sprout outside .claude raises PathTraversalError."""
        from goopy.blob import PathTraversalError

        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.FACT, summary="Escape")
            # Subdir with parent traversal should raise
            with pytest.raises(PathTraversalError):
                glob.sprout(blob, "test", subdir="../escape")

    def test_decompose_twice(self):
        """Decomposing same blob twice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))
            blob = Blob(type=BlobType.CONTEXT, summary="Decompose me")
            glob.sprout(blob, "victim")

            glob.decompose("victim")
            glob.decompose("victim")  # Should not crash

    def test_get_while_writing(self):
        """Concurrent read during write (simulated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            for i in range(100):
                blob = Blob(type=BlobType.FACT, summary=f"Version {i}")
                glob.sprout(blob, "contested")
                retrieved = glob.get("contested")
                # Should get a valid blob

    def test_list_with_non_xml_files(self):
        """List when .claude has non-XML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create valid blob
            blob = Blob(type=BlobType.FACT, summary="Valid")
            glob.sprout(blob, "valid")

            # Create non-XML files
            (Path(tmpdir) / ".claude" / "readme.md").write_text("# Readme")
            (Path(tmpdir) / ".claude" / "notes.txt").write_text("Notes")
            (Path(tmpdir) / ".claude" / "fake.blob.xml.bak").write_text("backup")

            blobs = glob.list_blobs()
            assert len(blobs) == 1

    def test_surface_with_corrupted_blob(self):
        """Surface when one blob is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create valid blob
            blob = Blob(type=BlobType.CONSTRAINT, summary="Valid", scope=BlobScope.ALWAYS)
            glob.sprout(blob, "valid", subdir="constraints")

            # Create corrupted file
            (Path(tmpdir) / ".claude" / "constraints" / "corrupted.blob.xml").write_text("garbage")

            # Should not crash
            relevant = glob.surface_relevant()
            assert len(relevant) >= 1

    def test_migrate_partially_written_file(self):
        """Migrate with partially written file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create a truncated XML
            (Path(tmpdir) / ".claude" / "partial.blob.xml").write_text(
                '<blob type="fact" scope="project" v="1"><sum'
            )

            # Should not crash
            outdated = glob.check_migrations()

    def test_zero_byte_blob_file(self):
        """Zero-byte blob file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            (Path(tmpdir) / ".claude" / "empty.blob.xml").write_text("")

            # Should not crash
            blobs = glob.list_blobs()
            # Empty file should be skipped

    def test_blob_file_is_directory(self):
        """What if a .blob.xml path is actually a directory?"""
        with tempfile.TemporaryDirectory() as tmpdir:
            glob = Glob(Path(tmpdir))

            # Create a valid blob
            blob = Blob(type=BlobType.FACT, summary="Valid")
            glob.sprout(blob, "valid")

            # Create a directory with blob extension (weird but possible)
            (Path(tmpdir) / ".claude" / "fake.blob.xml").mkdir(parents=True, exist_ok=True)

            # Should handle gracefully
            blobs = glob.list_blobs()


class TestAbsurdDatetimes:
    """Absurd datetime handling."""

    def test_blob_from_future(self):
        """Blob with future timestamp."""
        future = datetime(2099, 12, 31)
        blob = Blob(type=BlobType.FACT, summary="Future", updated=future)
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert restored.updated.year == 2099

    def test_blob_from_past(self):
        """Blob with ancient timestamp."""
        past = datetime(1970, 1, 1)
        blob = Blob(type=BlobType.FACT, summary="Ancient", updated=past)
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert restored.updated.year == 1970

    def test_blob_y2k_date(self):
        """Y2K edge case date."""
        y2k = datetime(2000, 1, 1)
        blob = Blob(type=BlobType.FACT, summary="Y2K", updated=y2k)
        xml = blob.to_xml()
        restored = Blob.from_xml(xml)
        assert restored.updated.year == 2000

    def test_malformed_date_in_xml(self):
        """Malformed date string in XML."""
        xml = '<blob type="fact" scope="project" v="2" updated="not-a-date"><summary>Test</summary></blob>'
        with pytest.raises(ValueError):
            Blob.from_xml(xml)

    def test_invalid_date_format(self):
        """Date in wrong format."""
        xml = '<blob type="fact" scope="project" v="2" updated="01/15/2025"><summary>Test</summary></blob>'
        with pytest.raises(ValueError):
            Blob.from_xml(xml)


class TestAbsurdVersions:
    """Absurd version handling."""

    def test_negative_version(self):
        """Negative version number."""
        blob = Blob(type=BlobType.FACT, summary="Negative", version=-1)
        assert blob.needs_migration()  # -1 < BLOB_VERSION

    def test_huge_version(self):
        """Extremely large version number."""
        blob = Blob(type=BlobType.FACT, summary="Future proof", version=9999999)
        assert not blob.needs_migration()  # Already "newer" than current

    def test_zero_version(self):
        """Zero version."""
        blob = Blob(type=BlobType.FACT, summary="Zero", version=0)
        assert blob.needs_migration()

    def test_non_integer_version_in_xml(self):
        """Non-integer version in XML attribute."""
        xml = '<blob type="fact" scope="project" v="abc" updated="2025-01-01"><summary>Test</summary></blob>'
        with pytest.raises(ValueError):
            Blob.from_xml(xml)

    def test_float_version_in_xml(self):
        """Float version in XML attribute."""
        xml = '<blob type="fact" scope="project" v="1.5" updated="2025-01-01"><summary>Test</summary></blob>'
        with pytest.raises(ValueError):
            Blob.from_xml(xml)
