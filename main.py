import time
import sys
from apriori import Apriori
from fpTree import FPGrowth
from visualization import AlgorithmVisualizer, visualize_apriori_results, visualize_fpgrowth_results, visualize_rules


def print_header(title):
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")


def run_apriori(min_sup, min_conf):
    print_header("APRIORI ALGORITHM")
    
    apriori = Apriori(min_sup=min_sup, min_conf=min_conf)
    
    start_time = time.time()
    itemsets = apriori.run_apriori()
    execution_time = time.time() - start_time
    
    rules = None
    if itemsets:
        rules = apriori.generate_rules(itemsets)
        
        print_header("APRIORI VISUALIZATIONS")
        print("Generating visualizations for Apriori...")
        visualize_apriori_results(apriori)
        
        if rules:
            visualize_rules(rules, algorithm_name="Apriori")
            print("✓ Apriori visualizations saved successfully!")
    
    return apriori, itemsets, rules, execution_time


def run_fpgrowth(min_sup, min_conf):
    print_header("FP-GROWTH ALGORITHM")
    
    fpgrowth = FPGrowth(min_sup=min_sup, min_conf=min_conf)
    
    start_time = time.time()
    itemsets = fpgrowth.run_fpgrowth()
    execution_time = time.time() - start_time
    
    rules = None
    if itemsets:
        rules = fpgrowth.generate_rules(itemsets)
        
        print_header("FP-GROWTH VISUALIZATIONS")
        print("Generating visualizations for FP-Growth...")
        visualize_fpgrowth_results(fpgrowth, itemsets)
        
        if rules:
            visualize_rules(rules, algorithm_name="FP-Growth")
            print("✓ FP-Growth visualizations saved successfully!")
    
    return fpgrowth, itemsets, rules, execution_time


def compare_algorithms(apriori_obj, apriori_itemsets, apriori_time, 
                      fpgrowth_itemsets, fpgrowth_time):
    print_header("ALGORITHM COMPARISON")
    
    visualizer = AlgorithmVisualizer()
    
    flat_apriori = {k: v for d in apriori_obj.all_frequent_itemsets.values() for k, v in d.items()}
    
    print("Generating comparison visualizations...")
    visualizer.compare_algorithms(
        apriori_itemsets=flat_apriori,
        fpgrowth_itemsets=fpgrowth_itemsets,
        apriori_time=apriori_time,
        fpgrowth_time=fpgrowth_time
    )
    print("✓ Comparison visualizations saved successfully!")


def print_summary(apriori_itemsets, apriori_rules, apriori_time,
                 fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time):
    print_header("EXECUTION SUMMARY")
    
    print(f"{'Metric':<30} {'Apriori':>20} {'FP-Growth':>20}")
    print("-" * 72)
    
    apriori_total = sum(len(items) for items in apriori_itemsets.values()) if apriori_itemsets else 0
    fpgrowth_total = len(fpgrowth_itemsets) if fpgrowth_itemsets else 0
    
    print(f"{'Total Frequent Itemsets':<30} {apriori_total:>20} {fpgrowth_total:>20}")
    print(f"{'Association Rules Generated':<30} {len(apriori_rules) if apriori_rules else 0:>20} {len(fpgrowth_rules) if fpgrowth_rules else 0:>20}")
    print(f"{'Execution Time (seconds)':<30} {apriori_time:>20.4f} {fpgrowth_time:>20.4f}")
    
    if apriori_time > 0 and fpgrowth_time > 0:
        speedup = apriori_time / fpgrowth_time
        faster = "FP-Growth" if speedup > 1 else "Apriori"
        factor = speedup if speedup > 1 else 1/speedup
        print(f"\n{faster} is {factor:.2f}x faster")
    
    print("\n" + "="*72)
    
    print("\n📊 Generated Visualization Files:")
    print("  • algorithm_comparison.png")
    print("  • apriori_itemset_distribution.png")
    print("  • apriori_top_itemsets.png")
    print("  • apriori_association_rules.png")
    print("  • apriori_rules_network.png")
    print("  • fp-growth_itemset_distribution.png")
    print("  • fp-growth_top_itemsets.png")
    print("  • fp-growth_association_rules.png")
    print("  • fp-growth_rules_network.png")
    print("  • fp_tree_structure.png")


def main():
    MIN_SUP_COUNT = 200
    MIN_CONFIDENCE = 0.3
    RUN_APRIORI = False
    RUN_FPGROWTH = True
    
    print_header("FREQUENT PATTERN MINING & ASSOCIATION RULE LEARNING")
    print(f"Configuration:")
    print(f"  • Minimum Support Count: {MIN_SUP_COUNT}")
    print(f"  • Minimum Confidence: {MIN_CONFIDENCE * 100}%")
    print(f"  • Run Apriori: {RUN_APRIORI}")
    print(f"  • Run FP-Growth: {RUN_FPGROWTH}")
    print(f"\n💡 Tip: Higher support count = faster execution but fewer patterns.")
    print(f"   Adjust MIN_SUP_COUNT in main.py (try 200, 300, or 500).")
    
    try:
        apriori_obj, apriori_itemsets, apriori_rules, apriori_time = None, None, None, 0
        
        if RUN_APRIORI:
            apriori_obj, apriori_itemsets, apriori_rules, apriori_time = run_apriori(
                MIN_SUP_COUNT, MIN_CONFIDENCE
            )
        else:
            print_header("APRIORI ALGORITHM - SKIPPED")
            print("Set RUN_APRIORI = True in main.py to enable Apriori.")
        
        fpgrowth_obj, fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time = None, None, None, 0
        
        if RUN_FPGROWTH:
            fpgrowth_obj, fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time = run_fpgrowth(
                MIN_SUP_COUNT, MIN_CONFIDENCE
            )
        else:
            print_header("FP-GROWTH ALGORITHM - SKIPPED")
            print("Set RUN_FPGROWTH = True in main.py to enable FP-Growth.")

        
        if apriori_itemsets and fpgrowth_itemsets:
            compare_algorithms(
                apriori_obj, apriori_itemsets, apriori_time,
                fpgrowth_itemsets, fpgrowth_time
            )
        
        if apriori_itemsets or fpgrowth_itemsets:
            print_summary(
                apriori_itemsets, apriori_rules, apriori_time,
                fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time
            )
        
        print_header("✓ ALL OPERATIONS COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
