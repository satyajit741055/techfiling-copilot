import json
from pathlib import Path

enriched_path = Path("goldens/golden_retriever_dataset_v3_enriched.jsonl")

# Get q051 and q053
for line in enriched_path.open(encoding="utf-8"):
    case = json.loads(line)
    if case["id"] in ["q051", "q053"]:
        print(f"=== {case['id']} ===")
        print(f"Query: {case['query']}")
        print(f"Doc ID: {case.get('doc_id')}")
        orders = [rel["value"] for rel in case["relevant"]]
        print(f"Element orders: {orders}")
        text = case.get("relevant_text", "")
        print(f"Relevant text length: {len(text)} chars")
        print(f"Relevant text (first 500 chars):\n{text[:500]}...")
        print()
