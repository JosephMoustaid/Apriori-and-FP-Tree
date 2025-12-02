import pandas as pd
import numpy as np
import math
import sys

class Apriori():
    
    def __init__(self, min_sup):
        self.df = pd.read_csv('transformed_dataset.csv')
        self.min_sup = min_sup
        # set the constants 
        self.N = len(self.df)
        self.W = self.df.apply(lambda x: x.astype(str).str.len().max())
        
        # usefull
        self.unique_1_items = self.df['ProductName'].explode().unique().tolist()

        self.d = len(self.unique_1_items)
        self.possible_itemsets = 2 ** self.d- 1
        self.possible_associations = 3 ** self.d - 2 ** (self.d -1 ) + 1
        sys.set_int_max_str_digits(0)
        print('Starting Apriori Algorithm')
        print('d = ' + str(self.d))
        print('W = ' + str(self.W))
        print('N = ' + str(self.N))
        print('min sup = ' + str(self.min_sup))
        print('Total number of possible item sets is : ' + str(self.possible_itemsets ))
        print('Total number of possible assoications is : ' + str(self.possible_associations ))

    def L1_generation(self):
        # flatter the column of lists into a series of individuals 
        exploded_series = self.df['ProductName'].explode()
        C1_counts = exploded_series.value_counts()
        
        frequent_items = C1_counts[C1_counts >= self.min_sup]
        print("frequent itmes are : " )
        print(frequent_items)
        return frequent_items
    def candidate_generation(self, LK_minus_1_itemsets, k ):
        Ck = set()
        n = len(LK_minus_1_itemsets)
        
        for i in range(n):
            for j in range(i+1, n ):
                t1 = LK_minus_1_temsets[i]
                t2 = LK_minus_1_itemsets[j]
                if t1[:-1] == t2[:-1]:
                    new_itemset = t1 + (t2[-1],)
                    
                    # prune check 
                    if(self._prune_check(new_itemset, LK_minus_1_itemsets, k):
                       Ck.add(new_itemset)
        return Ck
    def _prune_check(self, candidate, Lk_minus_1_itemsets, k):
    """Checks if all (k-1) subsets of the candidate are present in Lk_minus_1."""
    
    Lk_minus_1_set = set(Lk_minus_1_itemsets)
                       
    # For every subset (formed by removing one element) of the candidate
    for i in range(k):
        # Create a subset (k-1 tuple) by excluding the element at index i
        subset = candidate[:i] + candidate[i+1:]
        
        # If ANY subset is NOT found in Lk_minus_1, the candidate is infrequent and must be pruned
        if subset not in Lk_minus_1_set:
            return False
            
    return True # All subsets are frequent, so the candidate is keptdef 


if __name__ == "__main__" :
    MIN_SUP = 4
    MIN_CONF = 0.75
    apriori = Apriori(min_sup=MIN_SUP)
    apriori.L1_generation()
