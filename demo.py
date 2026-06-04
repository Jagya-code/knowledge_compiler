"""
demo.py — Run the full pipeline using the mock LLM (no Azure credentials needed).

Usage:
    python demo.py                          # uses sample docs + guidance
    python demo.py --no-guidance            # AI-only mode (no user guidance)
    python demo.py --docs path1 path2 ...   # custom documents

Output: output/hierarchy.json  (and output/hierarchy_no_guidance.json for comparison)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Activate mock BEFORE importing pipeline modules
os.environ["USE_MOCK_LLM"] = "1"
import mock_llm
mock_llm.activate()

# Now import the pipeline
from main import run_pipeline


def print_tree(nodes: list, indent: int = 0):
    for node in nodes:
        prefix = "  " * indent
        tag = "[user]" if node.get("user_defined") else "[ai]  "
        print(f"{prefix}{tag} [{node['node_id']}] {node['title']}")
        print(f"{prefix}       docs: {', '.join(node.get('source_docs', []))}")
        print(f"{prefix}       {node['summary'][:80]}...")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 2)


def main():
    parser = argparse.ArgumentParser(description="Document Hierarchy Builder Demo")
    parser.add_argument("--docs", nargs="+", help="Paths to input documents")
    parser.add_argument("--guidance", default="sample_guidance.json", help="Path to guidance JSON/Excel")
    parser.add_argument("--no-guidance", action="store_true", help="Run in AI-only mode")
    parser.add_argument("--output", default="output/hierarchy.json", help="Output path")
    args = parser.parse_args()

    doc_dir = Path("sample_docs")
    doc_paths = args.docs or [str(p) for p in sorted(doc_dir.glob("*.txt"))]

    if not doc_paths:
        print("No documents found. Place .txt files in sample_docs/ or pass --docs.")
        sys.exit(1)

    guidance_path = None if args.no_guidance else args.guidance
    mode_label = "AI-ONLY" if args.no_guidance else "HYBRID (AI + User Guidance)"

    print(f"\n{'='*60}")
    print(f"  Document Hierarchy Builder — {mode_label}")
    print(f"{'='*60}")
    print(f"  Documents : {len(doc_paths)}")
    print(f"  Guidance  : {guidance_path or 'none'}")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}\n")

    tree = run_pipeline(
        doc_paths=doc_paths,
        guidance_path=guidance_path,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"  RESULT — {len(tree)} top-level topics")
    print(f"{'='*60}\n")
    print_tree(tree)

    # Also save a pretty-printed version for easy reading
    pretty_path = Path(args.output).with_suffix(".pretty.json")
    with open(pretty_path, "w") as f:
        json.dump(tree, f, indent=2)
    print(f"\n  Full tree saved to: {args.output}")
    print(f"  Pretty copy saved to: {pretty_path}")

    # If running with guidance, also generate AI-only for comparison
    if not args.no_guidance:
        print(f"\n{'='*60}")
        print("  Running AI-ONLY mode for comparison...")
        print(f"{'='*60}\n")

        # Re-activate mock (was not cleared)
        mock_llm.activate()
        from main import run_pipeline as rp2
        ai_tree = rp2(
            doc_paths=doc_paths,
            guidance_path=None,
            output_path="output/hierarchy_ai_only.json",
        )
        print(f"\n  AI-only tree saved to: output/hierarchy_ai_only.json")
        print(f"\n  Compare the two outputs to see guidance influence.")


if __name__ == "__main__":
    main()
