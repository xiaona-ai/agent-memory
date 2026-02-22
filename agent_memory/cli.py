"""CLI entry point."""
import argparse
import sys
from . import store
from .config import load_config


def main():
    config = load_config()

    parser = argparse.ArgumentParser(prog="agent-memory", description="Lightweight memory store for AI agents")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize .agent-memory/ in current directory")

    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--meta", default="", help="key=value metadata pairs, comma-separated")

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", nargs="?", default="", help="Search query")
    p_search.add_argument("-n", type=int, default=None, help="Max results (default: from config)")
    p_search.add_argument("--tag", help="Filter by tag")

    p_list = sub.add_parser("list", help="List recent memories")
    p_list.add_argument("-n", type=int, default=20, help="Number of entries")

    p_export = sub.add_parser("export", help="Export memories")
    p_export.add_argument(
        "--format",
        choices=["md", "json"],
        default=None,
        dest="fmt",
        help=f"Export format (default: {config.get('default_export_format', 'md')})",
    )

    p_delete = sub.add_parser("delete", help="Delete a memory by ID")
    p_delete.add_argument("id", help="Memory ID to delete")
    p_delete.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    p_tag = sub.add_parser("tag", help="Manage tags on a memory")
    p_tag.add_argument("id", help="Memory ID")
    p_tag.add_argument("--add", dest="add_tags", default="", help="Comma-separated tags to add")
    p_tag.add_argument("--remove", dest="remove_tags", default="", help="Comma-separated tags to remove")

    sub.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "init":
        store.init_store()
    elif args.command == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        meta = {}
        if args.meta:
            for pair in args.meta.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    meta[k.strip()] = v.strip()
        store.add_memory(args.text, tags=tags, metadata=meta)
    elif args.command == "search":
        max_results = args.n if args.n is not None else config.get("max_results", 10)
        if args.tag and not args.query:
            entries = store.list_memories(n=9999)
            results = [e for e in entries if args.tag in e.get("tags", [])][:max_results]
        elif args.tag:
            results = store.search_memories(args.query, limit=9999)
            results = [e for e in results if args.tag in e.get("tags", [])][:max_results]
        else:
            if not args.query:
                print("Error: query is required unless --tag is specified.")
                sys.exit(1)
            results = store.search_memories(args.query, limit=max_results)
        _print_entries(results)
    elif args.command == "list":
        entries = store.list_memories(n=args.n)
        _print_entries(entries)
    elif args.command == "export":
        fmt = args.fmt or config.get("default_export_format", "md")
        print(store.export_memories(fmt))
    elif args.command == "delete":
        if not args.force:
            confirm = input(f"Delete memory {args.id}? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return
        if store.delete_memory(args.id):
            print(f"Deleted memory {args.id}")
        else:
            print(f"Memory {args.id} not found.")
            sys.exit(1)
    elif args.command == "tag":
        add_tags = [t.strip() for t in args.add_tags.split(",") if t.strip()] if args.add_tags else []
        remove_tags = [t.strip() for t in args.remove_tags.split(",") if t.strip()] if args.remove_tags else []
        if not add_tags and not remove_tags:
            print("Error: specify --add or --remove tags.")
            sys.exit(1)
        result = store.tag_memory(args.id, add_tags=add_tags, remove_tags=remove_tags)
        if result:
            print(f"Updated tags for {args.id}: {', '.join(result['tags']) or '(none)'}")
        else:
            print(f"Memory {args.id} not found.")
            sys.exit(1)
    elif args.command == "config":
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


def _print_entries(entries):
    if not entries:
        print("No memories found.")
        return
    for e in entries:
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        tags = f" [{', '.join(e['tags'])}]" if e.get("tags") else ""
        print(f"[{e['id']}] {ts}{tags}")
        print(f"  {e['text']}")
        print()


if __name__ == "__main__":
    main()
