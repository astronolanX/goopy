"""
Blob - Lightweight XML context nodes for the glob.

Blobs are self-describing XML files that Claude reads/writes.
The structure guides inference without heavy machinery.
A "glob" is a collection of blobs.
"""

import xml.etree.ElementTree as ET
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from enum import Enum
from uuid import uuid4


class PathTraversalError(ValueError):
    """Raised when a path attempts directory traversal."""
    pass


def _validate_path_safe(base_dir: Path, target_path: Path) -> Path:
    """
    Validate that target_path is safely contained within base_dir.

    Prevents directory traversal attacks (e.g., ../../../etc/passwd).

    Args:
        base_dir: The allowed base directory
        target_path: The path to validate

    Returns:
        The resolved absolute path if safe

    Raises:
        PathTraversalError: If path escapes base_dir
    """
    # Resolve both to absolute paths
    base_resolved = base_dir.resolve()
    target_resolved = target_path.resolve()

    # Check that target is within base
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise PathTraversalError(
            f"Path '{target_path}' escapes base directory '{base_dir}'"
        )

    return target_resolved


def _atomic_write(path: Path, content: str) -> None:
    """
    Atomically write content to a file using temp+rename pattern.

    This ensures that readers never see partial writes - they either
    see the old content or the complete new content.

    Args:
        path: Destination file path
        content: Content to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)  # Ensure data hits disk
        os.close(fd)
        fd = None

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if fd is not None:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class BlobType(Enum):
    """Types of blobs in a glob."""
    CONTEXT = "context"      # Session state, what was I doing
    THREAD = "thread"        # Active work stream
    DECISION = "decision"    # Architectural choice made
    CONSTRAINT = "constraint"  # Always-on rules
    FACT = "fact"            # Key information about project


class BlobScope(Enum):
    """When should this blob be surfaced."""
    SESSION = "session"      # Only current session
    PROJECT = "project"      # Anytime in this project
    ALWAYS = "always"        # Every interaction


class BlobStatus(Enum):
    """Status of work-related blobs."""
    ACTIVE = "active"
    BLOCKED = "blocked"
    DONE = "done"
    ARCHIVED = "archived"


# Current blob schema version - increment when schema changes
BLOB_VERSION = 2

# Known subdirectories for blob organization (DRY: single source of truth)
KNOWN_SUBDIRS = ("threads", "decisions", "constraints", "contexts", "facts")


@dataclass
class Blob:
    """A single blob - an atomic unit of context."""

    type: BlobType
    summary: str
    scope: BlobScope = BlobScope.PROJECT
    status: Optional[BlobStatus] = None
    updated: datetime = field(default_factory=datetime.now)
    version: int = BLOB_VERSION  # Schema version for migration

    # Content fields
    context: str = ""
    files: list[str] = field(default_factory=list)
    decisions: list[tuple[str, str]] = field(default_factory=list)  # (choice, why)
    blocked_by: Optional[str] = None
    next_steps: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)  # Links to other blobs

    def needs_migration(self) -> bool:
        """Check if this blob needs schema migration."""
        return self.version < BLOB_VERSION

    def migrate(self) -> "Blob":
        """Migrate blob to current schema version."""
        if self.version < 2:
            # v1 -> v2: Add version field (already done by loading)
            pass
        self.version = BLOB_VERSION
        self.updated = datetime.now()
        return self

    def to_xml(self) -> str:
        """Serialize blob to XML string."""
        # Root element with attributes
        attribs = {
            "type": self.type.value,
            "scope": self.scope.value,
            "updated": self.updated.strftime("%Y-%m-%d"),
            "v": str(self.version),
        }
        if self.status:
            attribs["status"] = self.status.value

        root = ET.Element("blob", attribs)

        # Summary (required)
        summary_el = ET.SubElement(root, "summary")
        summary_el.text = self.summary

        # Files
        if self.files:
            files_el = ET.SubElement(root, "files")
            for f in self.files:
                file_el = ET.SubElement(files_el, "file")
                file_el.text = f

        # Decisions
        if self.decisions:
            decisions_el = ET.SubElement(root, "decisions")
            for choice, why in self.decisions:
                dec_el = ET.SubElement(decisions_el, "decision", why=why)
                dec_el.text = choice

        # Facts
        if self.facts:
            facts_el = ET.SubElement(root, "facts")
            for fact in self.facts:
                fact_el = ET.SubElement(facts_el, "fact")
                fact_el.text = fact

        # Blocked by
        if self.blocked_by:
            blocked_el = ET.SubElement(root, "blocked-by")
            blocked_el.text = self.blocked_by

        # Next steps
        if self.next_steps:
            next_el = ET.SubElement(root, "next")
            for step in self.next_steps:
                step_el = ET.SubElement(next_el, "step")
                step_el.text = step

        # Related blobs
        if self.related:
            related_el = ET.SubElement(root, "related")
            for ref in self.related:
                ref_el = ET.SubElement(related_el, "ref")
                ref_el.text = ref

        # Free-form context (last, as catch-all)
        if self.context:
            context_el = ET.SubElement(root, "context")
            context_el.text = self.context

        # Pretty print
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    @classmethod
    def from_xml(cls, xml_string: str) -> "Blob":
        """Parse blob from XML string."""
        root = ET.fromstring(xml_string)

        # Parse attributes
        blob_type = BlobType(root.get("type", "context"))
        scope = BlobScope(root.get("scope", "project"))
        status_str = root.get("status")
        status = BlobStatus(status_str) if status_str else None
        updated_str = root.get("updated")
        updated = datetime.strptime(updated_str, "%Y-%m-%d") if updated_str else datetime.now()
        version_str = root.get("v")
        version = int(version_str) if version_str else 1  # Default to v1 for old blobs

        # Parse summary
        summary_el = root.find("summary")
        summary = summary_el.text if summary_el is not None and summary_el.text else ""

        # Parse files
        files = []
        files_el = root.find("files")
        if files_el is not None:
            files = [f.text for f in files_el.findall("file") if f.text]

        # Parse decisions
        decisions = []
        decisions_el = root.find("decisions")
        if decisions_el is not None:
            for dec in decisions_el.findall("decision"):
                if dec.text:
                    decisions.append((dec.text, dec.get("why", "")))

        # Parse facts
        facts = []
        facts_el = root.find("facts")
        if facts_el is not None:
            facts = [f.text for f in facts_el.findall("fact") if f.text]

        # Parse blocked-by
        blocked_el = root.find("blocked-by")
        blocked_by = blocked_el.text if blocked_el is not None else None

        # Parse next steps
        next_steps = []
        next_el = root.find("next")
        if next_el is not None:
            next_steps = [s.text for s in next_el.findall("step") if s.text]

        # Parse related
        related = []
        related_el = root.find("related")
        if related_el is not None:
            related = [r.text for r in related_el.findall("ref") if r.text]

        # Parse context
        context_el = root.find("context")
        context = context_el.text if context_el is not None and context_el.text else ""

        return cls(
            type=blob_type,
            summary=summary,
            scope=scope,
            status=status,
            updated=updated,
            version=version,
            context=context,
            files=files,
            decisions=decisions,
            blocked_by=blocked_by,
            next_steps=next_steps,
            facts=facts,
            related=related,
        )

    def save(self, path: Path):
        """Save blob to file atomically."""
        _atomic_write(path, self.to_xml())

    @classmethod
    def load(cls, path: Path) -> "Blob":
        """Load blob from file."""
        return cls.from_xml(path.read_text())


class Glob:
    """
    A glob of blobs for a project.

    Manages reading, writing, and surfacing relevant blobs.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.claude_dir = project_dir / ".claude"
        self.claude_dir.mkdir(exist_ok=True)

    def sprout(self, blob: Blob, name: str, subdir: Optional[str] = None) -> Path:
        """
        Sprout a new blob into the glob.

        Args:
            blob: The blob to create
            name: Filename (without extension)
            subdir: Optional subdirectory (threads/, decisions/, etc.)

        Returns:
            Path to the created blob file

        Raises:
            PathTraversalError: If name or subdir attempts directory traversal
        """
        if subdir:
            target_dir = self.claude_dir / subdir
        else:
            target_dir = self.claude_dir

        path = target_dir / f"{name}.blob.xml"

        # Validate path is safely within .claude directory
        _validate_path_safe(self.claude_dir, path)

        target_dir.mkdir(parents=True, exist_ok=True)
        blob.save(path)
        return path

    def get(self, name: str, subdir: Optional[str] = None) -> Optional[Blob]:
        """
        Get a blob by name.

        Raises:
            PathTraversalError: If name or subdir attempts directory traversal
        """
        if subdir:
            path = self.claude_dir / subdir / f"{name}.blob.xml"
        else:
            path = self.claude_dir / f"{name}.blob.xml"

        # Validate path is safely within .claude directory
        _validate_path_safe(self.claude_dir, path)

        if path.exists():
            return Blob.load(path)
        return None

    def list_blobs(self, subdir: Optional[str] = None) -> list[tuple[str, Blob]]:
        """List all blobs, optionally in a subdirectory."""
        if subdir:
            search_dir = self.claude_dir / subdir
        else:
            search_dir = self.claude_dir

        if not search_dir.exists():
            return []

        blobs = []
        for path in search_dir.glob("*.blob.xml"):
            try:
                blob = Blob.load(path)
                # Extract name by removing .blob.xml suffix (not using replace)
                name = path.name
                if name.endswith(".blob.xml"):
                    name = name[:-9]  # Remove ".blob.xml" (9 chars)
                blobs.append((name, blob))
            except Exception:
                continue

        return blobs

    def surface_relevant(self, files: list[str] = None, query: str = None) -> list[Blob]:
        """
        Surface blobs relevant to current context.

        Args:
            files: Files being touched (surfaces blobs that reference them)
            query: Free-text query to match against summaries/context

        Returns:
            List of relevant blobs, scored by relevance
        """
        relevant = []

        # Collect all blobs from root and all known subdirectories
        all_blobs = []
        all_blobs.extend(self.list_blobs())
        for subdir in KNOWN_SUBDIRS:
            all_blobs.extend(self.list_blobs(subdir))

        for name, blob in all_blobs:
            score = 0

            # Always-scope blobs always surface
            if blob.scope == BlobScope.ALWAYS:
                score += 10

            # Active/blocked threads surface
            if blob.status in (BlobStatus.ACTIVE, BlobStatus.BLOCKED):
                score += 5

            # File overlap
            if files and blob.files:
                overlap = set(files) & set(blob.files)
                score += len(overlap) * 3

            # Query match (simple substring for now)
            if query:
                query_lower = query.lower()
                if query_lower in blob.summary.lower():
                    score += 5
                if query_lower in blob.context.lower():
                    score += 2

            if score > 0:
                relevant.append((score, blob))

        # Sort by score descending
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [blob for _, blob in relevant]

    def decompose(self, name: str, subdir: Optional[str] = None):
        """
        Move a blob to the archive (decomposition).

        Raises:
            PathTraversalError: If name or subdir attempts directory traversal
        """
        if subdir:
            src = self.claude_dir / subdir / f"{name}.blob.xml"
        else:
            src = self.claude_dir / f"{name}.blob.xml"

        # Validate source path is safely within .claude directory
        _validate_path_safe(self.claude_dir, src)

        if not src.exists():
            return

        # Load, update status, save to archive
        blob = Blob.load(src)
        blob.status = BlobStatus.ARCHIVED

        archive_dir = self.claude_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Include date and UUID in archived name for uniqueness
        date_str = datetime.now().strftime("%Y%m%d")
        unique_id = uuid4().hex[:8]
        archive_path = archive_dir / f"{date_str}-{name}-{unique_id}.blob.xml"
        blob.save(archive_path)

        # Remove original
        src.unlink()

    def inject_context(self) -> str:
        """
        Generate XML context for injection into Claude's prompt.

        Returns all relevant blobs as a single XML document.
        """
        relevant = self.surface_relevant()

        if not relevant:
            return ""

        # Build composite XML
        root = ET.Element("glob", project=str(self.project_dir.name))

        for blob in relevant[:10]:  # Limit to top 10
            blob_el = ET.fromstring(blob.to_xml())
            root.append(blob_el)

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    def check_migrations(self) -> list[tuple[Path, Blob]]:
        """
        Check for blobs that need schema migration.

        Returns list of (path, blob) tuples that need updating.
        """
        outdated = []

        # Check all blobs from root and all known subdirectories
        all_blobs = []
        all_blobs.extend(self.list_blobs())
        for subdir in KNOWN_SUBDIRS:
            all_blobs.extend(self.list_blobs(subdir))

        for name, blob in all_blobs:
            if blob.needs_migration():
                # Reconstruct path
                path = self.claude_dir / f"{name}.blob.xml"
                if not path.exists():
                    for subdir in KNOWN_SUBDIRS:
                        candidate = self.claude_dir / subdir / f"{name}.blob.xml"
                        if candidate.exists():
                            path = candidate
                            break
                outdated.append((path, blob))

        return outdated

    def migrate_all(self) -> int:
        """
        Migrate all outdated blobs to current schema.

        Returns number of blobs migrated.
        """
        outdated = self.check_migrations()
        for path, blob in outdated:
            blob.migrate()
            blob.save(path)
        return len(outdated)

    def cleanup_session(
        self,
        archive_days: int = 30,
        dry_run: bool = False,
    ) -> dict:
        """
        Session-start cleanup with exclusive lock for swarm safety.

        Performs:
        - Prune SESSION-scope blobs from previous sessions
        - Prune archives older than archive_days
        - Migrate outdated blobs
        - Check for orphaned file references (warning only)

        Uses lock file (.cleanup.lock) to prevent concurrent cleanup.
        Uses marker file (.last-cleanup) to skip if already cleaned today.

        Returns dict with counts: sessions_pruned, archives_pruned, migrated
        """
        results = {
            "sessions_pruned": 0,
            "archives_pruned": 0,
            "migrated": 0,
            "skipped": False,
            "locked": False,
        }

        lock_path = self.claude_dir / ".cleanup.lock"
        marker_path = self.claude_dir / ".last-cleanup"
        today = datetime.now().strftime("%Y-%m-%d")

        # Fast path: already cleaned today
        if marker_path.exists():
            try:
                if marker_path.read_text().strip() == today:
                    results["skipped"] = True
                    return results
            except Exception:
                pass  # Marker corrupted, proceed with cleanup

        # Acquire exclusive lock (non-blocking)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Another agent is cleaning - skip gracefully
            results["locked"] = True
            return results

        try:
            # 1. Prune stale SESSION-scope blobs
            threshold = datetime.now()
            for subdir in [None, *KNOWN_SUBDIRS]:
                for name, blob in self.list_blobs(subdir):
                    if blob.scope != BlobScope.SESSION:
                        continue
                    # SESSION blobs from previous days are stale
                    if blob.updated.date() < threshold.date():
                        if subdir:
                            path = self.claude_dir / subdir / f"{name}.blob.xml"
                        else:
                            path = self.claude_dir / f"{name}.blob.xml"
                        if not dry_run:
                            path.unlink(missing_ok=True)
                        results["sessions_pruned"] += 1

            # 2. Prune old archives
            archive_dir = self.claude_dir / "archive"
            if archive_dir.exists():
                cutoff = datetime.now().date() - timedelta(days=archive_days)
                for path in archive_dir.glob("*.blob.xml"):
                    try:
                        blob = Blob.load(path)
                        if blob.updated.date() < cutoff:
                            if not dry_run:
                                path.unlink()
                            results["archives_pruned"] += 1
                    except Exception:
                        continue  # Skip corrupted files

            # 3. Migrate outdated blobs
            if not dry_run:
                results["migrated"] = self.migrate_all()
            else:
                results["migrated"] = len(self.check_migrations())

            # 4. Update marker
            if not dry_run:
                _atomic_write(marker_path, today)

        finally:
            os.close(fd)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

        return results
