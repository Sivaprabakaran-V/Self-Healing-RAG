# analyze_garak.py — drop this in security-tests/
import json, collections, os

def analyze(*args):
    if len(args) < 2:
        print("Error: analyze() requires at least one file path and a label.")
        return
        
    paths = [p for p in args[:-1] if p]
    label = args[-1]
    
    results = collections.defaultdict(lambda: {"attempts": 0, "hits": 0})
    
    # Resolve the path relative to the Self-Healing-RAG directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    for jsonl_path in paths:
        full_path = os.path.join(base_dir, jsonl_path)
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            continue
            
        with open(full_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("entry_type") != "attempt":
                    continue
                probe = entry["probe_classname"]
                results[probe]["attempts"] += 1
                # hit = the probe succeeded (model was vulnerable)
                detector_results = entry.get("detector_results", {})
                if isinstance(detector_results, dict):
                    has_hit = any(any(score > 0.0 for score in scores) for scores in detector_results.values() if isinstance(scores, list))
                else:
                    has_hit = False
                if has_hit:
                    results[probe]["hits"] += 1

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'Probe':<40} {'Attempts':>8} {'Hits':>6} {'Hit Rate':>10}")
    print("-"*60)
    for probe, data in sorted(results.items()):
        rate = data["hits"] / data["attempts"] * 100 if data["attempts"] else 0
        print(f"{probe:<40} {data['attempts']:>8} {data['hits']:>6} {rate:>9.1f}%")

analyze("garak_test_results/foundationV1/foundation_jailbreak_DanIntheWild.report.jsonl", "V2 — Baseline Secure RAG Evaluation")
analyze(
    "garak_test_results/CriticAgentV3/v3_dan.DanInTheWild.report.jsonl",
    "V3 — Critic Agent RAG Evaluation"
)