import pandas as pd
import numpy as np
import math
import sys
import time
from itertools import combinations

class Apriori():
    
    def __init__(self, min_sup, min_conf):
        # Time tracking start
        start_time = time.time()
        
        self.df = pd.read_csv('transformed_dataset.csv')
        self.min_sup = min_sup
        self.min_conf = min_conf
        
        # --- Constants ---
        self.N = len(self.df)
        # Calculate max string length for display (not used in algorithm logic)
        self.W = self.df.apply(lambda x: x.astype(str).str.len().max()).max() 
        
        # Use a list of sets for fast transaction lookups later
        self.transactions = [set(item_list) for item_list in self.df['ProductIDList']]
        
        # Initial item population
        self.unique_1_items = self.df['ProductIDList'].explode().unique().tolist()
        self.d = len(self.unique_1_items)
        
        # Use Python power operator for arbitrary precision
        try:
            # Set integer conversion limit to unlimited (0) for printing large numbers
            sys.set_int_max_str_digits(0) 
            self.possible_itemsets = (2 ** self.d) - 1
            self.possible_associations = (3 ** self.d) - (2 ** (self.d + 1)) + 1
        except Exception:
            # Fallback if setting the limit is blocked or problematic
            self.possible_itemsets = f"2^{self.d} (too large to display)"
            self.possible_associations = f"3^{self.d} (too large to display)"
            
        
        # --- Printing Program Info ---
        print('\n--- Starting Apriori Algorithm Setup ---')
        print(f'Total Execution Time: {time.time() - start_time:.4f} seconds (Setup Only)')
        print(f'Total Transactions (N): {self.N}')
        print(f'Total Unique Items (d): {self.d}')
        print(f'Min Support Count: {self.min_sup} (Count)')
        print(f'Min Confidence: {self.min_conf * 100}%')
        print(f'Total possible item sets: {self.possible_itemsets}')
        print(f'Total possible associations: {self.possible_associations}')
        print('--------------------------------------\n')


    def L1_generation(self):
        """Generates the frequent 1-itemset (L1) by filtering C1 by min_sup."""
        
        # Flatten the column into a series of individual items (Candidate C1 counts)
        C1_counts = self.df['ProductName'].explode().value_counts()
        
        # Filter for frequent items (L1)
        # Convert index (item names) to tuple, which is hashable for later steps
        frequent_items_series = C1_counts[C1_counts >= self.min_sup]
        L1 = {(item,): count for item, count in frequent_items_series.items()}
        
        print(f"L1 generated. Found {len(L1)} frequent 1-itemsets.")
        return L1

    def _prune_check(self, candidate, Lk_minus_1_itemsets, k):
        """Checks if all (k-1) subsets of the candidate are frequent (i.e., in Lk_minus_1)."""
        
        Lk_minus_1_set = set(Lk_minus_1_itemsets)
                        
        # For every subset (formed by removing one element) of the candidate
        for i in range(k):
            # Create a subset (k-1 tuple) by excluding the element at index i
            subset = candidate[:i] + candidate[i+1:]
            
            # If ANY subset is NOT found in Lk_minus_1, prune (return False)
            if subset not in Lk_minus_1_set:
                return False
                
        return True # All subsets are frequent, so the candidate is kept


    def candidate_generation(self, Lk_minus_1, k):
        """Generates candidate k-itemsets (Ck) from Lk-1 using join and prune."""
        
        Lk_minus_1_itemsets = sorted(list(Lk_minus_1.keys()))
        Ck = set()
        n = len(Lk_minus_1_itemsets)
        
        # 1. Self-Join Step
        for i in range(n):
            for j in range(i + 1, n):
                t1 = Lk_minus_1_itemsets[i]
                t2 = Lk_minus_1_itemsets[j]
                
                # Join condition: check if the first k-2 elements are identical
                if t1[:-1] == t2[:-1]:
                    new_itemset = t1 + (t2[-1],) # Creates a new tuple of size k
                    
                    # 2. Pruning Step (only needed for k >= 3, but safe to run always)
                    if self._prune_check(new_itemset, Lk_minus_1_itemsets, k):
                        Ck.add(new_itemset)
                        
        return Ck
        

    def frequent_itemset_search(self, Ck):
        """Scans the database to count support for Ck and returns Lk."""
        
        Lk_counts = {}
        
        # 1. Efficiently check containment for all candidates
        # This is the most computationally expensive step, optimized by using sets
        for candidate in Ck:
            count = 0
            candidate_set = set(candidate)
            
            # Use the pre-converted list of transaction sets (self.transactions)
            for transaction_set in self.transactions:
                if candidate_set.issubset(transaction_set):
                    count += 1
            
            # 2. Filter by min_sup
            if count >= self.min_sup:
                Lk_counts[candidate] = count
                
        return Lk_counts


    def run_apriori(self):
        """Driver method to run the iterative Apriori process."""
        
        start_time = time.time()
        
        # Dictionary to store all frequent itemsets {size_k: {itemset_tuple: support_count}}
        self.all_frequent_itemsets = {} 
        
        # 1. L1 Generation (k=1)
        Lk_minus_1 = self.L1_generation()
        self.all_frequent_itemsets[1] = Lk_minus_1
        
        if not Lk_minus_1:
            print("Apriori stopped: L1 is empty.")
            return None
            
        k = 2
        
        # 2. Iterative Lk Generation (k >= 2)
        while True:
            # Generate candidate k-itemsets Ck from Lk-1
            Ck = self.candidate_generation(Lk_minus_1, k) 
            
            if not Ck:
                print(f"Stop condition: C{k} is empty.")
                break
                
            # Search for frequent itemsets Lk from Ck
            Lk = self.frequent_itemset_search(Ck)
            
            if not Lk:
                print(f"Stop condition: L{k} is empty.")
                break
            
            self.all_frequent_itemsets[k] = Lk
            Lk_minus_1 = Lk 
            k += 1

        end_time = time.time()
        print(f"\n--- Apriori Iteration Complete ---")
        print(f"Found frequent itemsets up to size k={k-1}. Total time: {end_time - start_time:.4f} seconds.")
        
        return self.all_frequent_itemsets


    def generate_rules(self, all_frequent_itemsets):
        """Generates and prints association rules based on min_conf."""
        
        rules = []
        
        # Iterate through all frequent itemsets Lk where k >= 2
        for k, Lk in all_frequent_itemsets.items():
            if k < 2:
                continue
            
            for itemset, support_k in Lk.items():
                
                # For each itemset, generate all possible non-empty subsets (antecedents)
                for i in range(1, k):
                    for antecedent_tuple in combinations(itemset, i):
                        
                        antecedent_set = set(antecedent_tuple)
                        consequent_set = set(itemset) - antecedent_set
                        consequent_tuple = tuple(sorted(list(consequent_set)))
                        
                        # Get the support of the antecedent (A)
                        # The support for A is in Lk_minus_1 (or L_size_i)
                        antecedent_support = self.all_frequent_itemsets[len(antecedent_tuple)].get(antecedent_tuple, 0)
                        
                        if antecedent_support == 0:
                            continue

                        # Calculate Confidence
                        confidence = support_k / antecedent_support
                        
                        if confidence >= self.min_conf:
                            # Calculate Lift for interestingness (optional but standard)
                            # Support(B) = Support(Consequent)
                            consequent_support = self.all_frequent_itemsets[len(consequent_tuple)].get(consequent_tuple, 0)
                            
                            # Support(B) / N is the probability of B
                            lift = confidence / (consequent_support / self.N)
                            
                            rules.append({
                                'antecedent': antecedent_tuple,
                                'consequent': consequent_tuple,
                                'support': support_k / self.N, # Normalized support
                                'confidence': confidence,
                                'lift': lift
                            })

        # --- Print Rules ---
        print(f"\n--- Association Rules (Confidence >= {self.min_conf*100}%) ---")
        if not rules:
            print("No strong association rules found.")
            return

        rules_df = pd.DataFrame(rules)
        # Sort by lift or confidence for best results
        rules_df = rules_df.sort_values(by=['lift', 'confidence'], ascending=False)
        
        # Format the output nicely
        rules_df['Rule'] = rules_df.apply(
            lambda row: f'{set(row["antecedent"])} -> {set(row["consequent"])}', axis=1
        )
        print(rules_df[['Rule', 'support', 'confidence', 'lift']].head(10))
        print(f"\nTotal rules generated: {len(rules)}")


if __name__ == "__main__" :
    # Configuration Parameters
    MIN_SUP_COUNT = 2 # Absolute count threshold (e.g., must appear 4 times)
    MIN_CONFIDENCE = -0.5    
    # 1. Initialize and Run
    apriori = Apriori(min_sup=MIN_SUP_COUNT, min_conf=MIN_CONFIDENCE)
    
    # Run the main itemset generation loop
    all_frequent_itemsets = apriori.run_apriori()
    
    # 2. Generate and print the association rules
    if all_frequent_itemsets:
        apriori.generate_rules(all_frequent_itemsets)
