import pandas as pd
import sys
import time
from collections import defaultdict, namedtuple
from itertools import combinations

# Named tuple for clear representation of frequent patterns
Pattern = namedtuple('Pattern', ['itemset', 'support'])

class FPTreeNode:
    """A node in the Frequent Pattern Tree."""
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {} # Dictionary of item: FPTreeNode
        self.node_link = None # Link to the next node with the same item

    def increment(self, count):
        self.count += count

class FPGrowth:
    
    def __init__(self, min_sup, min_conf=0.0):
        start_time = time.time()
        
        self.df = pd.read_csv('transformed_dataset.csv')
        self.min_sup = min_sup
        self.min_conf = min_conf
        
        # --- Constants ---
        self.N = len(self.df)
        self.unique_items = self.df['ProductIDList'].explode().unique().tolist()
        self.d = len(self.unique_items)
        
        # FP-Tree specific attributes
        self.header_table = {}
        self.tree_root = FPTreeNode('null')
        self.frequent_patterns = []
        
        # Print info (suppressing large numbers using try-except structure)
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
        """Step 1: Calculate item support and create a header table sorted by frequency."""
        
        item_counts = self.df['ProductName'].explode().value_counts()
        
        # Filter by min_sup to get L1 items
        L1_items = item_counts[item_counts >= self.min_sup]
        
        # Sort items by frequency (descending)
        # Store item, count, and initialize node link to None
        self.header_table = {
            item: {'count': count, 'node_link': None}
            for item, count in L1_items.items()
        }
        
        # Get the sorted list of frequent item names
        self.frequent_items_sorted = list(self.header_table.keys())

        if not self.header_table:
            print("No frequent items found (L1 is empty).")
            return False
        return True


    def _insert_transaction(self, transaction, current_node):
        """Inserts a single transaction into the FP-Tree."""
        
        if not transaction:
            return
            
        # Get the first item in the (frequency-sorted) transaction
        item = transaction[0]
        
        # 1. Check if item exists in the current node's children
        if item in current_node.children:
            child = current_node.children[item]
            child.increment(1)
        else:
            # 2. Create new node and link it
            child = FPTreeNode(item, count=1, parent=current_node)
            current_node.children[item] = child
            
            # 3. Update Header Table node link
            self._update_header_link(child)
            
        # 4. Recursively insert the rest of the transaction
        self._insert_transaction(transaction[1:], child)


    def _update_header_link(self, node):
        """Adds the new node to the linked list chain in the header table."""
        item = node.item
        
        if self.header_table[item]['node_link'] is None:
            # First node for this item
            self.header_table[item]['node_link'] = node
        else:
            # Traverse the node link list to the end and append the new node
            current = self.header_table[item]['node_link']
            while current.node_link is not None:
                current = current.node_link
            current.node_link = node


    def _build_fp_tree(self):
        """Step 2: Build the complete FP-Tree."""
        
        # 1. Map transactions to frequency-sorted items
        for transaction_list in self.df['ProductName']:
            # Filter transaction items to only include frequent items (L1)
            # Sort the transaction by the frequency order defined in the header table
            ordered_transaction = [
                item for item in self.frequent_items_sorted if item in transaction_list
            ]
            
            # Insert the sorted, filtered transaction into the tree
            self._insert_transaction(ordered_transaction, self.tree_root)
        
        print("FP-Tree construction complete.")


    def _mine_frequent_patterns(self, header_table, base_itemset, min_sup):
        """Step 3: Recursively mine frequent patterns from the FP-Tree structure."""
        
        # Iterate over items in the header table, from least frequent to most frequent
        # Sorting by reverse frequency helps find the longest patterns first
        sorted_items = sorted(
            header_table.keys(), 
            key=lambda item: header_table[item]['count'],
            reverse=False
        )

        for item in sorted_items:
            support = header_table[item]['count']
            
            # 1. Generate the new frequent pattern (item + base_itemset)
            new_pattern = tuple(sorted(base_itemset + (item,)))
            self.frequent_patterns.append(Pattern(new_pattern, support))
            
            # 2. Build Conditional Pattern Base (CPB) for the current item
            conditional_pattern_base = []
            
            # Follow the node links for the current item
            current_node = header_table[item]['node_link']
            while current_node is not None:
                # Find the path to the root (excluding the current item and the root)
                path = []
                parent = current_node.parent
                while parent.item != 'null':
                    path.append(parent.item)
                    parent = parent.parent
                
                # Add the path to the CPB, weighted by the node's count
                if path:
                    conditional_pattern_base.append((tuple(reversed(path)), current_node.count))
                    
                current_node = current_node.node_link

            # 3. Build Conditional FP-Tree (CFT)
            
            # Collect frequencies for the conditional prefix paths
            conditional_header_table = defaultdict(int)
            
            for path, count in conditional_pattern_base:
                for prefix_item in path:
                    conditional_header_table[prefix_item] += count

            # Filter the conditional header table by min_sup
            new_header_table = {
                k: {'count': v, 'node_link': None} 
                for k, v in conditional_header_table.items() 
                if v >= min_sup
            }
            
            # If the new conditional table is not empty, recurse
            if new_header_table:
                # Create a new, temporary conditional tree root
                conditional_root = FPTreeNode('null')
                
                # Sort the path for deterministic insertion
                sorted_conditional_items = sorted(
                    new_header_table.keys(), 
                    key=lambda i: new_header_table[i]['count'], 
                    reverse=True
                )
                
                # Insert paths into the temporary conditional tree
                for path, count in conditional_pattern_base:
                    # Filter and reorder path based on the new header table's frequency order
                    ordered_conditional_path = [
                        item for item in sorted_conditional_items if item in path
                    ]
                    
                    # Insert the path 'count' times
                    for _ in range(count):
                        self._insert_transaction(ordered_conditional_path, conditional_root)

                # Recursively mine the conditional tree
                self._mine_frequent_patterns(new_header_table, new_pattern, min_sup)


    def run_fpgrowth(self):
        """Driver method to run the entire FP-Growth process."""
        
        start_time = time.time()

        # Step 1: Build Header Table
        if not self._build_header_table():
            return

        # Step 2: Build FP-Tree
        self._build_fp_tree()
        
        # Step 3: Mine Frequent Patterns
        print("Starting recursive FP-Tree mining...")
        self._mine_frequent_patterns(self.header_table, tuple(), self.min_sup)

        end_time = time.time()
        print(f"\n--- FP-Growth Algorithm Complete ---")
        print(f"Total time: {end_time - start_time:.4f} seconds.")
        
        # L1 is missing from the recursive mining, so add them back
        for item, data in self.header_table.items():
            self.frequent_patterns.append(Pattern((item,), data['count']))
            
        # Filter duplicates (e.g., L1 items added twice) and remove patterns with low support 
        # (Though L1 is filtered, recursive calls might add patterns that are duplicates)
        final_patterns = {}
        for pattern in self.frequent_patterns:
            itemset = pattern.itemset
            support = pattern.support
            if itemset not in final_patterns or support > final_patterns[itemset]:
                final_patterns[itemset] = support
        
        print(f"Total unique frequent itemsets found: {len(final_patterns)}")
        return final_patterns


    def generate_rules(self, frequent_itemsets):
        """Generates and prints association rules based on min_conf."""
        
        # Convert dictionary to be easier to work with
        itemset_supports = {k: v for k, v in frequent_itemsets.items()}
        rules = []
        
        print(f"\n--- Association Rules (Confidence >= {self.min_conf*100}%) ---")
        
        # Iterate only on itemsets with size k >= 2
        for itemset, support_itemset in itemset_supports.items():
            if len(itemset) < 2:
                continue

            # Generate all non-empty subsets of the frequent itemset
            for i in range(1, len(itemset)):
                for antecedent_tuple in combinations(itemset, i):
                    
                    antecedent_tuple = tuple(sorted(antecedent_tuple))
                    antecedent_support = itemset_supports.get(antecedent_tuple, 0)
                    
                    # Prevent division by zero
                    if antecedent_support == 0:
                        continue 
                        
                    # Calculate Confidence
                    confidence = support_itemset / antecedent_support
                    
                    if confidence >= self.min_conf:
                        consequent_set = set(itemset) - set(antecedent_tuple)
                        consequent_tuple = tuple(sorted(list(consequent_set)))
                        
                        # Calculate Lift
                        consequent_support = itemset_supports.get(consequent_tuple, 0)
                        
                        # Use a small number to prevent division by zero in lift calculation if support is 0
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

        # Use pandas for presentation
        rules_df = pd.DataFrame(rules)
        rules_df = rules_df.sort_values(by=['lift', 'confidence'], ascending=False)
        
        rules_df['Rule'] = rules_df.apply(
            lambda row: f'{set(row["antecedent"])} -> {set(row["consequent"])}', axis=1
        )
        
        # Print top 10 rules
        print(rules_df[['Rule', 'support', 'confidence', 'lift']].head(10))
        print(f"\nTotal rules generated: {len(rules)}")


if __name__ == "__main__" :
    # Configuration Parameters
    MIN_SUP_COUNT = 2
    MIN_CONFIDENCE = 0.6 
    
    # 1. Initialize and Run
    fpgrowth = FPGrowth(min_sup=MIN_SUP_COUNT, min_conf=MIN_CONFIDENCE)
    
    # Run the main itemset generation loop
    all_frequent_itemsets = fpgrowth.run_fpgrowth()
    
    # 2. Generate and print the association rules
    if all_frequent_itemsets:
        fpgrowth.generate_rules(all_frequent_itemsets)
