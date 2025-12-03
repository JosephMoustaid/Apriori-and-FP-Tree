import pandas as pd
import sys
import time
from collections import defaultdict, namedtuple
from itertools import combinations

Pattern = namedtuple('Pattern', ['itemset', 'support'])

class FPTreeNode:
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment(self, count):
        self.count += count

class FPGrowth:
    
    def __init__(self, min_sup, min_conf=0.0):
        start_time = time.time()
        
        self.df = pd.read_csv('transformed_dataset.csv')
        self.df['ProductIDList'] = self.df['ProductIDList'].apply(eval)
        self.df['ProductName'] = self.df['ProductName'].apply(eval)
        
        self.min_sup = min_sup
        self.min_conf = min_conf
        
        self.N = len(self.df)
        self.unique_items = self.df['ProductIDList'].explode().unique().tolist()
        self.d = len(self.unique_items)
        
        self.header_table = {}
        self.tree_root = FPTreeNode('null')
        self.frequent_patterns = []
        
        try:
            sys.set_int_max_str_digits(0) 
            self.possible_itemsets = (2 ** self.d) - 1
        except Exception:
            self.possible_itemsets = f"2^{self.d} (too large to display)"
            
        print('\n--- Starting FP-Growth Algorithm Setup ---')
        print(f'Total Execution Time: {time.time() - start_time:.4f} seconds (Setup Only)')
        print(f'Total Transactions (N): {self.N}')
        print(f'Total Unique Items (d): {self.d}')
        print(f'Min Support Count: {self.min_sup} (Count)')
        print(f'Min Confidence: {self.min_conf * 100}%')
        print(f'Total possible item sets: {self.possible_itemsets}')
        print('--------------------------------------\n')


    def _build_header_table(self):
        item_counts = self.df['ProductName'].explode().value_counts()
        
        L1_items = item_counts[item_counts >= self.min_sup]
        
        self.header_table = {
            item: {'count': count, 'node_link': None}
            for item, count in L1_items.items()
        }
        
        self.frequent_items_sorted = list(self.header_table.keys())

        if not self.header_table:
            print("No frequent items found (L1 is empty).")
            return False
        return True


    def _insert_transaction(self, transaction, current_node):
        if not transaction:
            return
            
        item = transaction[0]
        
        if item in current_node.children:
            child = current_node.children[item]
            child.increment(1)
        else:
            child = FPTreeNode(item, count=1, parent=current_node)
            current_node.children[item] = child
            
            self._update_header_link(child)
            
        self._insert_transaction(transaction[1:], child)


    def _update_header_link(self, node):
        item = node.item
        
        if self.header_table[item]['node_link'] is None:
            self.header_table[item]['node_link'] = node
        else:
            current = self.header_table[item]['node_link']
            while current.node_link is not None:
                current = current.node_link
            current.node_link = node


    def _build_fp_tree(self):
        for transaction_list in self.df['ProductName']:
            ordered_transaction = [
                item for item in self.frequent_items_sorted if item in transaction_list
            ]
            
            self._insert_transaction(ordered_transaction, self.tree_root)
        
        print("FP-Tree construction complete.")


    def _mine_frequent_patterns(self, header_table, base_itemset, min_sup, max_depth=5, current_depth=0):
        if current_depth >= max_depth:
            return
            
        sorted_items = sorted(
            header_table.keys(), 
            key=lambda item: header_table[item]['count'],
            reverse=False
        )

        for item in sorted_items:
            support = header_table[item]['count']
            
            new_pattern = tuple(sorted(base_itemset + (item,)))
            self.frequent_patterns.append(Pattern(new_pattern, support))
            
            conditional_pattern_base = []
            
            current_node = header_table[item]['node_link']
            while current_node is not None:
                path = []
                parent = current_node.parent
                while parent.item != 'null':
                    path.append(parent.item)
                    parent = parent.parent
                
                if path:
                    conditional_pattern_base.append((tuple(reversed(path)), current_node.count))
                    
                current_node = current_node.node_link

            conditional_header_table = defaultdict(int)
            
            for path, count in conditional_pattern_base:
                for prefix_item in path:
                    conditional_header_table[prefix_item] += count

            new_header_table = {
                k: {'count': v, 'node_link': None} 
                for k, v in conditional_header_table.items() 
                if v >= min_sup
            }
            
            if new_header_table:
                self._mine_frequent_patterns(new_header_table, new_pattern, min_sup, max_depth, current_depth + 1)


    def run_fpgrowth(self):
        start_time = time.time()

        if not self._build_header_table():
            return

        self._build_fp_tree()
        
        print("Starting recursive FP-Tree mining (limited to depth 5)...")
        self._mine_frequent_patterns(self.header_table, tuple(), self.min_sup, max_depth=5)

        end_time = time.time()
        print(f"\n--- FP-Growth Algorithm Complete ---")
        print(f"Total time: {end_time - start_time:.4f} seconds.")
        
        for item, data in self.header_table.items():
            self.frequent_patterns.append(Pattern((item,), data['count']))
            
        final_patterns = {}
        for pattern in self.frequent_patterns:
            itemset = pattern.itemset
            support = pattern.support
            if itemset not in final_patterns or support > final_patterns[itemset]:
                final_patterns[itemset] = support
        
        print(f"Total unique frequent itemsets found: {len(final_patterns)}")
        return final_patterns


    def generate_rules(self, frequent_itemsets):
        itemset_supports = {k: v for k, v in frequent_itemsets.items()}
        rules = []
        
        print(f"\n--- Association Rules (Confidence >= {self.min_conf*100}%) ---")
        
        for itemset, support_itemset in itemset_supports.items():
            if len(itemset) < 2:
                continue

            for i in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, i):
                    
                    antecedent_tuple = tuple(sorted(antecedent_tuple))
                    antecedent_support = itemset_supports.get(antecedent_tuple, 0)
                    
                    if antecedent_support == 0:
                        continue 
                        
                    confidence = support_itemset / antecedent_support
                    
                    if confidence >= self.min_conf:
                        consequent_set = set(itemset) - set(antecedent_tuple)
                        consequent_tuple = tuple(sorted(list(consequent_set)))
                        
                        consequent_support = itemset_supports.get(consequent_tuple, 0)
                        
                        prob_consequent = consequent_support / self.N if consequent_support else 1e-10 
                        lift = confidence / prob_consequent
                        
                        rules.append({
                            'antecedent': antecedent_tuple,
                            'consequent': consequent_tuple,
                            'support_count': support_itemset,
                            'support': support_itemset / self.N, 
                            'confidence': confidence,
                            'lift': lift
                        })

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
