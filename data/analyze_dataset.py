import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

df = pd.read_csv('transformed_dataset.csv')

print("="*80)
print("DATASET ANALYSIS")
print("="*80)

print(f"\nBasic Statistics:")
print(f"  • Total Transactions: {len(df)}")

all_items = []
for item_list in df['ProductIDList']:
    all_items.extend(eval(item_list))

print(f"  • Total Items (with duplicates): {len(all_items)}")
print(f"  • Unique Items: {len(set(all_items))}")
print(f"  • Average Items per Transaction: {len(all_items) / len(df):.2f}")

transaction_sizes = df['ProductIDList'].apply(lambda x: len(eval(x)))
print(f"\nTransaction Size Distribution:")
print(f"  • Min: {transaction_sizes.min()}")
print(f"  • Max: {transaction_sizes.max()}")
print(f"  • Mean: {transaction_sizes.mean():.2f}")
print(f"  • Median: {transaction_sizes.median():.0f}")

item_counts = Counter(all_items)
sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nItem Frequency Analysis:")
print(f"  • Top 10 Most Frequent Items:")
for i, (item, count) in enumerate(sorted_items[:10], 1):
    support_pct = (count / len(df)) * 100
    print(f"    {i}. Item {item}: appears in {count} transactions ({support_pct:.2f}%)")

print(f"\n  • Items appearing at least:")
for threshold in [50, 100, 150, 200, 300, 500]:
    count = sum(1 for _, freq in item_counts.items() if freq >= threshold)
    print(f"    - {threshold} times: {count} items")

print(f"\nFinding Co-occurring Item Pairs...")
pair_counts = Counter()
for item_list_str in df['ProductIDList']:
    items = eval(item_list_str)
    if len(items) >= 2:
        for pair in combinations(sorted(items), 2):
            pair_counts[pair] += 1

print(f"\n  • Total unique pairs found: {len(pair_counts)}")
print(f"  • Top 10 Most Frequent Pairs:")
for i, (pair, count) in enumerate(pair_counts.most_common(10), 1):
    support_pct = (count / len(df)) * 100
    print(f"    {i}. Items {pair[0]} & {pair[1]}: {count} times ({support_pct:.2f}%)")

print(f"\n  • Pairs appearing at least:")
for threshold in [10, 20, 30, 50, 100]:
    count = sum(1 for _, freq in pair_counts.items() if freq >= threshold)
    print(f"    - {threshold} times: {count} pairs")

print(f"\n" + "="*80)
print("RECOMMENDED PARAMETERS")
print("="*80)

total_trans = len(df)
print(f"\nBased on your dataset with {total_trans} transactions:\n")

recommendations = [
    (0.001, 0.3, "Very Low (Find rare but strong patterns)"),
    (0.005, 0.4, "Low (Good for sparse data)"),
    (0.01, 0.5, "Medium-Low (Balanced for retail)"),
    (0.02, 0.5, "Medium (Common patterns)"),
    (0.05, 0.6, "High (Very common patterns)")
]

for min_sup_pct, min_conf, desc in recommendations:
    min_sup_count = int(total_trans * min_sup_pct)
    items_above = sum(1 for _, freq in item_counts.items() if freq >= min_sup_count)
    pairs_above = sum(1 for _, freq in pair_counts.items() if freq >= min_sup_count)
    
    print(f"Option: {desc}")
    print(f"  MIN_SUP_COUNT = {min_sup_count} ({min_sup_pct*100:.1f}% of transactions)")
    print(f"  MIN_CONFIDENCE = {min_conf}")
    print(f"  → Expected: ~{items_above} frequent items, ~{pairs_above} frequent pairs\n")

print("="*80)
print("💡 SUGGESTED STARTING POINT:")
print("="*80)
optimal_sup = int(total_trans * 0.01)
print(f"  MIN_SUP_COUNT = {optimal_sup}  (1% of transactions)")
print(f"  MIN_CONFIDENCE = 0.4  (40%)")
print(f"\nThis should find meaningful patterns without being too restrictive.")
print("Adjust in main.py based on your needs.")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

freq_values = list(item_counts.values())
axes[0, 0].hist(freq_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Frequency (# of transactions)', fontweight='bold')
axes[0, 0].set_ylabel('Number of Items', fontweight='bold')
axes[0, 0].set_title('Item Frequency Distribution', fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(alpha=0.3)

top_items = sorted_items[:20]
items_labels = [f"Item {item}" for item, _ in top_items]
items_counts = [count for _, count in top_items]
axes[0, 1].barh(range(len(items_labels)), items_counts, color='coral', edgecolor='black')
axes[0, 1].set_yticks(range(len(items_labels)))
axes[0, 1].set_yticklabels(items_labels, fontsize=8)
axes[0, 1].set_xlabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Top 20 Most Frequent Items', fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

axes[1, 0].hist(transaction_sizes, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Transaction Size (# of items)', fontweight='bold')
axes[1, 0].set_ylabel('Number of Transactions', fontweight='bold')
axes[1, 0].set_title('Transaction Size Distribution', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

if pair_counts:
    pair_freq_values = list(pair_counts.values())
    axes[1, 1].hist(pair_freq_values, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Co-occurrence Frequency', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Item Pairs', fontweight='bold')
    axes[1, 1].set_title('Item Pair Co-occurrence Distribution', fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Analysis visualization saved as 'dataset_analysis.png'")
