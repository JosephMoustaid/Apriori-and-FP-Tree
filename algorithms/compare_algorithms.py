import time
from apriori import Apriori
from fpTree import FPGrowth
from visualization import AlgorithmVisualizer, visualize_rules

MIN_SUP_COUNT = 2
MIN_CONFIDENCE = 0.6

print("="*70)
print("RUNNING APRIORI ALGORITHM")
print("="*70)

apriori = Apriori(min_sup=MIN_SUP_COUNT, min_conf=MIN_CONFIDENCE)
apriori_start = time.time()
apriori_itemsets = apriori.run_apriori()
apriori_time = time.time() - apriori_start

apriori_rules = None
if apriori_itemsets:
    apriori_rules = apriori.generate_rules(apriori_itemsets)

print("\n" + "="*70)
print("RUNNING FP-GROWTH ALGORITHM")
print("="*70)

fpgrowth = FPGrowth(min_sup=MIN_SUP_COUNT, min_conf=MIN_CONFIDENCE)
fpgrowth_start = time.time()
fpgrowth_itemsets = fpgrowth.run_fpgrowth()
fpgrowth_time = time.time() - fpgrowth_start

fpgrowth_rules = None
if fpgrowth_itemsets:
    fpgrowth_rules = fpgrowth.generate_rules(fpgrowth_itemsets)

print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

visualizer = AlgorithmVisualizer()

if apriori_itemsets and fpgrowth_itemsets:
    print("\nComparing both algorithms...")
    visualizer.compare_algorithms(
        apriori_itemsets={k: v for d in apriori.all_frequent_itemsets.values() for k, v in d.items()},
        fpgrowth_itemsets=fpgrowth_itemsets,
        apriori_time=apriori_time,
        fpgrowth_time=fpgrowth_time
    )

if apriori_rules:
    print("\nVisualizing Apriori rules...")
    visualize_rules(apriori_rules, algorithm_name="Apriori")

if fpgrowth_rules:
    print("\nVisualizing FP-Growth rules...")
    visualize_rules(fpgrowth_rules, algorithm_name="FP-Growth")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated visualization files:")
print("  - algorithm_comparison.png")
print("  - apriori_association_rules.png")
print("  - apriori_rules_network.png")
print("  - fp-growth_association_rules.png")
print("  - fp-growth_rules_network.png")
