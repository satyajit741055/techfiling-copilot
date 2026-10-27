import json
from pathlib import Path

enriched_path = Path("goldens/golden_retriever_dataset_v3_enriched.jsonl")

# Get q051
for line in enriched_path.open(encoding="utf-8"):
    case = json.loads(line)
    if case["id"] == "q051":
        text = case.get("relevant_text", "")
        # Split by document tags
        sections = text.split("[AAPL_10-K_2025]")
        sections = [s for s in sections if s.strip()]

        print("=== q051 ===")
        print(f"Total sections: {len(sections)}")
        print()

        for i, section in enumerate(sections):
            print(f"Section {i + 1}: {len(section)} chars")
            # Show first 200 chars
            print(section[:200])
            print("---")
