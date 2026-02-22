"""CLI entry point."""
import argparse
import sys
from . import store


def main():
    parser = argparse.ArgumentParser(prog="agent-memory", description="Lightweight memory store for AI agents")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize .agent-memory/ in current directory")

    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--meta", default="", help="key=value metadata pairs, comma-separated")

    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", type=int, default=10, help="Max results")

    p_list = sub.add_parser("list", help="List recent memories")
    p_list.add_argument("-n", type=int, default=20, help="Number of entries")

    p_export = sub.add_parser("export", help="Export memories")
    p_export.add_argument("--format", choices=["md", "json"], default="md", dest="fmt")

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
        results = store.search_memories(args.query, limit=args.n)
        _print_entries(results)
    elif args.command == "list":
        entries = store.list_memories(n=args.n)
        _print_entries(entries)
    elif args.command == "export":
        if args.fmt == "md":
            print(store.export_markdown())
        else:
            print(store.export_json())
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
