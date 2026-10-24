import json
from pathlib import Path

chunk_dir = Path('data/chunks/2026-10-24_12-46-23')

# Load all chunks by doc_id
chunks_by_doc = {
    'AAPL_10-K_2025': [],
    'MSFT_10-K_2026': [],
    'TSLA_10-K_2025': [],
}

for chunk_file in chunk_dir.glob('*.chunks.jsonl'):
    doc_id = chunk_file.stem.replace('_10-K_', '_10-K_').replace('.chunks', '')
    for line in chunk_file.open(encoding='utf-8'):
        chunk = json.loads(line)
        chunks_by_doc[doc_id].append(chunk)

# Check q051 and q053 element_orders
golden_path = Path('goldens/golden_retriever_dataset_v3.jsonl')
for line in golden_path.open(encoding='utf-8'):
    case = json.loads(line)
    if case['id'] in ['q051', 'q053']:
        print(f"=== {case['id']} ===")
        print(f"Query: {case['query']}")
        print(f"Doc ID: {case.get('doc_id')}")

        for rel in case['relevant']:
            orders = rel['value']
            if isinstance(orders[0], list):
                # Multiple groups
                for group in orders:
                    print(f"\n  Group: {group}")
                    for doc_id, chunks in chunks_by_doc.items():
                        found = []
                        for chunk in chunks:
                            if any(order in chunk.get('element_orders', []) for order in group):
                                found.append(chunk['chunk_id'])
                        if found:
                            print(f"    {doc_id}: {len(found)} chunks found")
            else:
                # Single group
                print(f"\n  Orders: {orders}")
                for doc_id, chunks in chunks_by_doc.items():
                    found = []
                    for chunk in chunks:
                        if any(order in chunk.get('element_orders', []) for order in orders):
                            found.append(chunk['chunk_id'])
                    if found:
                        print(f"    {doc_id}: {len(found)} chunks found")
        print()
