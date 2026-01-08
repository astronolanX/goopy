"""
Goopy CLI - XML blob management for Claude Code session memory.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def cmd_sprout(args):
    """Create a new blob."""
    from goopy.blob import Glob, Blob, BlobType, BlobScope, BlobStatus, KNOWN_SUBDIRS

    project_dir = Path.cwd()
    glob = Glob(project_dir)

    # Parse type (context not allowed - auto-created by persist hook)
    try:
        blob_type = BlobType(args.type)
    except ValueError:
        valid = ", ".join(t.value for t in BlobType if t != BlobType.CONTEXT)
        print(f"Invalid type: {args.type}", file=sys.stderr)
        print(f"Valid types: {valid}", file=sys.stderr)
        sys.exit(1)

    if blob_type == BlobType.CONTEXT:
        print("context blobs are auto-created by the persist hook", file=sys.stderr)
        print("Use: thread, decision, constraint, or fact", file=sys.stderr)
        sys.exit(1)

    # Derive scope from type (no user choice - cleaner API)
    scope_map = {
        BlobType.CONSTRAINT: BlobScope.ALWAYS,
        BlobType.THREAD: BlobScope.PROJECT,
        BlobType.DECISION: BlobScope.PROJECT,
        BlobType.FACT: BlobScope.PROJECT,
    }
    scope = scope_map[blob_type]

    # Parse status (only for threads)
    status = None
    if args.status:
        if blob_type != BlobType.THREAD:
            print("--status only applies to thread blobs", file=sys.stderr)
            sys.exit(1)
        try:
            status = BlobStatus(args.status)
        except ValueError:
            valid = ", ".join(s.value for s in BlobStatus)
            print(f"Invalid status: {args.status}", file=sys.stderr)
            print(f"Valid statuses: {valid}", file=sys.stderr)
            sys.exit(1)
    elif blob_type == BlobType.THREAD:
        status = BlobStatus.ACTIVE  # default for threads

    # Create blob
    blob = Blob(
        type=blob_type,
        summary=args.summary,
        scope=scope,
        status=status,
    )

    # Determine subdirectory based on type
    subdir_map = {
        BlobType.THREAD: "threads",
        BlobType.DECISION: "decisions",
        BlobType.CONSTRAINT: "constraints",
        BlobType.CONTEXT: "contexts",
        BlobType.FACT: "facts",
    }
    subdir = args.dir or subdir_map.get(blob_type)

    # Generate name from summary (kebab-case, max 30 chars)
    name = args.name
    if not name:
        name = args.summary.lower()
        name = "".join(c if c.isalnum() or c == " " else "" for c in name)
        name = "-".join(name.split())[:30]

    # Write blob
    path = glob.sprout(blob, name, subdir)
    rel_path = path.relative_to(project_dir)
    print(f"Sprouted: {rel_path}")


def cmd_decompose(args):
    """Archive stale blobs."""
    from goopy.blob import Glob, BlobScope, KNOWN_SUBDIRS

    project_dir = Path.cwd()
    glob = Glob(project_dir)

    # Find stale session blobs
    days = args.days or 7
    threshold = datetime.now() - timedelta(days=days)

    stale: list[tuple[Path, str, object]] = []

    # Check root and all subdirs
    for subdir in [None, *KNOWN_SUBDIRS]:
        for name, blob in glob.list_blobs(subdir):
            # Only session-scoped blobs are candidates for decomposition
            if blob.scope != BlobScope.SESSION:
                continue
            if blob.updated < threshold:
                if subdir:
                    path = glob.claude_dir / subdir / f"{name}.blob.xml"
                else:
                    path = glob.claude_dir / f"{name}.blob.xml"
                stale.append((path, name, blob))

    if not stale:
        print(f"No stale session blobs (>{days} days old)")
        return

    print(f"Found {len(stale)} stale session blob(s):")
    for path, name, blob in stale:
        rel_path = path.relative_to(project_dir)
        age = (datetime.now() - blob.updated).days
        print(f"  {rel_path} ({age}d old)")
        print(f"    {blob.summary[:60]}")

    if args.dry_run:
        print("\nRun without --dry-run to archive")
        return

    # Archive by deleting (session blobs are ephemeral)
    print()
    for path, name, blob in stale:
        path.unlink()
        rel_path = path.relative_to(project_dir)
        print(f"Archived: {rel_path}")

    print(f"\nDecomposed {len(stale)} blob(s)")


def cmd_migrate(args):
    """Migrate blobs to current schema version."""
    from goopy.blob import Glob, BLOB_VERSION

    project_dir = Path.cwd()
    glob = Glob(project_dir)

    outdated = glob.check_migrations()

    if not outdated:
        print(f"All blobs are at current version (v{BLOB_VERSION})")
        return

    if args.dry_run:
        print(f"Found {len(outdated)} blob(s) needing migration:")
        for path, blob in outdated:
            print(f"  {path.name} (v{blob.version} -> v{BLOB_VERSION})")
        print("\nRun without --dry-run to migrate")
        return

    count = glob.migrate_all()
    print(f"Migrated {count} blob(s) to v{BLOB_VERSION}")


def cmd_list(args):
    """Show blob population health and diagnostics."""
    from goopy.blob import Glob, Blob, BlobType, BlobScope, BlobStatus, BLOB_VERSION, KNOWN_SUBDIRS

    project_dir = Path.cwd()
    glob = Glob(project_dir)

    # Collect all blobs with their paths
    all_blobs: list[tuple[Path, str, Blob]] = []

    # Root level blobs
    for name, blob in glob.list_blobs():
        path = glob.claude_dir / f"{name}.blob.xml"
        all_blobs.append((path, name, blob))

    # Subdirectory blobs
    for subdir in KNOWN_SUBDIRS:
        for name, blob in glob.list_blobs(subdir):
            path = glob.claude_dir / subdir / f"{name}.blob.xml"
            all_blobs.append((path, name, blob))

    if not all_blobs:
        print("No blobs found in .claude/")
        print("Blobs are XML context files that persist across sessions.")
        return

    # Aggregate stats
    type_counts = defaultdict(int)
    scope_counts = defaultdict(int)
    status_counts = defaultdict(int)
    version_counts = defaultdict(int)
    total_tokens = 0
    needs_migration = []
    stale_sessions = []
    active_threads = []
    all_files_referenced = []
    missing_files = []

    now = datetime.now()
    stale_threshold = now - timedelta(days=7)

    for path, name, blob in all_blobs:
        type_counts[blob.type.value] += 1
        scope_counts[blob.scope.value] += 1
        if blob.status:
            status_counts[blob.status.value] += 1
        version_counts[blob.version] += 1

        # Estimate tokens (~200 per blob + content)
        xml_len = len(blob.to_xml())
        tokens = max(200, xml_len // 4)  # Rough estimate
        total_tokens += tokens

        # Track migrations needed
        if blob.needs_migration():
            needs_migration.append((path, name, blob))

        # Track stale sessions
        if blob.scope == BlobScope.SESSION and blob.updated < stale_threshold:
            stale_sessions.append((name, blob))

        # Track active threads
        if blob.type == BlobType.THREAD and blob.status == BlobStatus.ACTIVE:
            active_threads.append((name, blob))

        # Track file references
        for f in blob.files:
            all_files_referenced.append(f)
            # Check if file exists (expand ~ and check)
            file_path = Path(f).expanduser()
            if not file_path.is_absolute():
                file_path = project_dir / f
            if not file_path.exists():
                missing_files.append(f)

    # Output
    project_name = project_dir.name
    print(f"Glob Health: {project_name}")
    print("=" * 45)
    print()

    # Population summary
    print(f"Population: {len(all_blobs)} blob(s) (~{total_tokens:,} tokens)")

    # Type breakdown
    type_str = "  "
    for t in ["constraint", "thread", "decision", "context", "fact"]:
        if type_counts[t]:
            type_str += f"{t}: {type_counts[t]}  "
    if type_str.strip():
        print(type_str)

    # Scope breakdown
    scope_str = "  "
    for s in ["always", "project", "session"]:
        if scope_counts[s]:
            scope_str += f"{s}: {scope_counts[s]}  "
    if scope_str.strip():
        print(scope_str)

    print()

    # Active threads
    if active_threads:
        print(f"Active Threads: {len(active_threads)}")
        for name, blob in active_threads[:3]:
            next_count = len(blob.next_steps)
            next_str = f" ({next_count} next steps)" if next_count else ""
            print(f"  -> {blob.summary[:50]}{next_str}")
        if len(active_threads) > 3:
            print(f"  ... and {len(active_threads) - 3} more")
        print()

    # Schema health
    if needs_migration:
        print(f"Schema: ! {len(needs_migration)} blob(s) need migration")
        print(f"  Run: goopy migrate")
    else:
        print(f"Schema: OK all v{BLOB_VERSION}")
    print()

    # Staleness
    if stale_sessions:
        print(f"Staleness: ! {len(stale_sessions)} session blob(s) >7 days old")
        for name, blob in stale_sessions[:2]:
            print(f"  -> {name} (updated {blob.updated.strftime('%Y-%m-%d')})")
    else:
        session_count = scope_counts.get("session", 0)
        if session_count:
            print(f"Staleness: OK {session_count} session blob(s) all recent")
    print()

    # File references
    if all_files_referenced:
        if missing_files:
            print(f"File References: ! {len(missing_files)}/{len(all_files_referenced)} missing")
            for f in missing_files[:3]:
                print(f"  -> {f}")
        else:
            print(f"File References: OK {len(all_files_referenced)} file(s) all exist")
        print()

    # Injection impact
    # Estimate ~0.5ms per 100 tokens + 30ms baseline
    est_ms = 30 + (total_tokens // 200)
    print(f"Injection Impact: ~{total_tokens:,} tokens / ~{est_ms}ms per session")
    print()

    # Suggestions
    suggestions = []

    if needs_migration:
        suggestions.append(f"Run `goopy migrate` to update {len(needs_migration)} blob(s)")

    for name, blob in stale_sessions:
        suggestions.append(f"Archive stale session '{name}' (updated {blob.updated.strftime('%b %d')})")

    for name, blob in active_threads:
        if len(blob.next_steps) > 5:
            suggestions.append(f"Thread '{name}' has {len(blob.next_steps)} next steps - consider splitting")

    if not type_counts.get("constraint"):
        suggestions.append("No constraints defined - consider adding project rules")

    if len(active_threads) > 3:
        suggestions.append(f"{len(active_threads)} active threads - consider completing some")

    if missing_files:
        suggestions.append(f"Update or remove {len(missing_files)} stale file reference(s)")

    if suggestions:
        print("Suggestions:")
        for s in suggestions[:5]:
            print(f"  -> {s}")
    else:
        print("Suggestions: None - glob looks healthy!")


def cmd_cleanup(args):
    """Session-start cleanup with swarm-safe locking."""
    from goopy.blob import Glob

    project_dir = Path.cwd()
    glob = Glob(project_dir)

    results = glob.cleanup_session(
        archive_days=args.archive_days or 30,
        dry_run=args.dry_run,
    )

    if results["skipped"]:
        print("Cleanup: Already cleaned today (skipped)")
        return

    if results["locked"]:
        print("Cleanup: Another agent is cleaning (skipped)")
        return

    total = results["sessions_pruned"] + results["archives_pruned"] + results["migrated"]

    if total == 0:
        print("Cleanup: Nothing to clean")
        return

    if args.dry_run:
        print("Cleanup (dry-run):")
    else:
        print("Cleanup:")

    if results["sessions_pruned"]:
        print(f"  Sessions pruned: {results['sessions_pruned']}")
    if results["archives_pruned"]:
        print(f"  Archives pruned: {results['archives_pruned']}")
    if results["migrated"]:
        print(f"  Blobs migrated: {results['migrated']}")

    if args.dry_run:
        print("\nRun without --dry-run to apply")


def main():
    parser = argparse.ArgumentParser(
        prog="goopy",
        description="XML blob system for Claude Code session memory",
    )
    parser.add_argument("--version", "-V", action="version", version="goopy 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sprout
    sprout_parser = subparsers.add_parser(
        "sprout",
        help="Create a new blob",
        description="Sprout a new blob into the glob (.claude/ directory)"
    )
    sprout_parser.add_argument("type", help="Blob type: thread, decision, constraint, fact")
    sprout_parser.add_argument("summary", help="Brief summary of the blob")
    sprout_parser.add_argument("--status", help="Status for threads: active, blocked, done, archived")
    sprout_parser.add_argument("--name", "-n", help="Blob filename (default: derived from summary)")
    sprout_parser.add_argument("--dir", "-d", help="Subdirectory override (default: based on type)")
    sprout_parser.set_defaults(func=cmd_sprout)

    # list
    list_parser = subparsers.add_parser(
        "list",
        help="Show blob health and diagnostics",
        description="Display population health and diagnostics for all blobs"
    )
    list_parser.set_defaults(func=cmd_list)

    # migrate
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate blobs to current schema",
        description="Upgrade blobs to the current schema version"
    )
    migrate_parser.add_argument("--dry-run", action="store_true", help="Preview migrations without applying")
    migrate_parser.set_defaults(func=cmd_migrate)

    # decompose
    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Archive stale session blobs",
        description="Find and archive session-scoped blobs older than threshold"
    )
    decompose_parser.add_argument("--days", type=int, help="Age threshold in days (default: 7)")
    decompose_parser.add_argument("--dry-run", action="store_true", help="Preview without archiving")
    decompose_parser.set_defaults(func=cmd_decompose)

    # cleanup
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Session-start cleanup (swarm-safe)",
        description="Prune stale sessions, old archives, and migrate blobs. Uses lock file for swarm safety."
    )
    cleanup_parser.add_argument("--archive-days", type=int, help="Days before pruning archives (default: 30)")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    cleanup_parser.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
