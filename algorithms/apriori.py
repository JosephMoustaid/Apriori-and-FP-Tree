import pandas as pd
import numpy as np
import math
import sys
import time
from itertools import combinations

class Apriori():
    
    def __init__(self, min_sup, min_conf):
        start_time = time.time()
        
        self.df = pd.read_csv('transformed_dataset.csv')
        self.df['ProductIDList'] = self.df['ProductIDList'].apply(eval)
        self.df['ProductName'] = self.df['ProductName'].apply(eval)
        
        self.min_sup = min_sup
        self.min_conf = min_conf
        
        self.N = len(self.df)
        self.W = self.df.apply(lambda x: x.astype(str).str.len().max()).max() 
        
        self.transactions = [set(item_list) for item_list in self.df['ProductIDList']]
        
        self.unique_1_items = self.df['ProductIDList'].explode().unique().tolist()
        self.d = len(self.unique_1_items)
        
        try:
            sys.set_int_max_str_digits(0) 
            self.possible_itemsets = (2 ** self.d) - 1
            self.possible_associations = (3 ** self.d) - (2 ** (self.d + 1)) + 1
        except Exception:
            self.possible_itemsets = f"2^{self.d} (too large to display)"
            self.possible_associations = f"3^{self.d} (too large to display)"
            
        
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
        C1_counts = self.df['ProductName'].explode().value_counts()
        
        frequent_items_series = C1_counts[C1_counts >= self.min_sup]
        L1 = {(item,): count for item, count in frequent_items_series.items()}
        
        print(f"L1 generated. Found {len(L1)} frequent 1-itemsets.")
        return L1

    def _prune_check(self, candidate, Lk_minus_1_itemsets, k):
        Lk_minus_1_set = set(Lk_minus_1_itemsets)
                        
        for i in range(k):
            subset = candidate[:i] + candidate[i+1:]
            
            if subset not in Lk_minus_1_set:
                return False
                
        return True


    def candidate_generation(self, Lk_minus_1, k):
        Lk_minus_1_itemsets = sorted(list(Lk_minus_1.keys()))
        Ck = set()
        n = len(Lk_minus_1_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                t1 = Lk_minus_1_itemsets[i]
                t2 = Lk_minus_1_itemsets[j]
                
                if t1[:-1] == t2[:-1]:
                    new_itemset = t1 + (t2[-1],)
                    
                    if self._prune_check(new_itemset, Lk_minus_1_itemsets, k):
                        Ck.add(new_itemset)
                        
        return Ck
        

    def frequent_itemset_search(self, Ck):
        Lk_counts = {}
        
        for candidate in Ck:
            count = 0
            candidate_set = set(candidate)
            
            for transaction_set in self.transactions:
                if candidate_set.issubset(transaction_set):
                    count += 1
            
            if count >= self.min_sup:
                Lk_counts[candidate] = count
                
        return Lk_counts


    def run_apriori(self):
        start_time = time.time()
        
        self.all_frequent_itemsets = {} 
        
        Lk_minus_1 = self.L1_generation()
        self.all_frequent_itemsets[1] = Lk_minus_1
        
        if not Lk_minus_1:
            print("Apriori stopped: L1 is empty.")
            return None
            
        k = 2
        
        while True:
            Ck = self.candidate_generation(Lk_minus_1, k) 
            
            if not Ck:
                print(f"Stop condition: C{k} is empty.")
                break
                
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
        rules = []
        
        for k, Lk in all_frequent_itemsets.items():
            if k < 2:
                continue
            
            for itemset, support_k in Lk.items():
                
                for i in range(1, k):
                    for antecedent_tuple in combinations(itemset, i):
                        
                        antecedent_set = set(antecedent_tuple)
                        consequent_set = set(itemset) - antecedent_set
                        consequent_tuple = tuple(sorted(list(consequent_set)))
                        
                        antecedent_support = self.all_frequent_itemsets[len(antecedent_tuple)].get(antecedent_tuple, 0)
                        
                        if antecedent_support == 0:
                            continue

                        confidence = support_k / antecedent_support
                        
                        if confidence >= self.min_conf:
                            consequent_support = self.all_frequent_itemsets[len(consequent_tuple)].get(consequent_tuple, 0)
                            
                            lift = confidence / (consequent_support / self.N)
                            
                            rules.append({
                                'antecedent': antecedent_tuple,
                                'consequent': consequent_tuple,
                                'support': support_k / self.N,
                                'confidence': confidence,
                                'lift': lift
                            })

        print(f"\n--- Association Rules (Confidence >= {self.min_conf*100}%) ---")
        if not rules:
            print("No strong association rules found.")
            return

        rules_df = pd.DataFrame(rules)
        rules_df = rules_df.sort_values(by=['lift', 'confidence'], ascending=False)
        
        rules_df['Rule'] = rules_df.apply(
            lambda row: f'{set(row["antecedent"])} -> {set(row["consequent"])}', axis=1
        )
        print(rules_df[['Rule', 'support', 'confidence', 'lift']].head(10))
        print(f"\nTotal rules generated: {len(rules)}")
        
        return rules
