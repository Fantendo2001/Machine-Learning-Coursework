class Apriori:
    def __init__(self, min_sup = 0.2):
        self.min_sup = min_sup # min support
        
    '''
    *Generating frequent 1-itemsets*
    in: data<pandas.core.frame.DataFrame>
    out: C<list>[<tuple>], D<dict>
    '''
    def _frequent_singleton(self, data):
        sup_count = data.sum()
        sup_count = sup_count[sup_count>=self.min_sup_count]
        sup_count.index = [(idx,) for idx in sup_count.index]
        C = sup_count.index
        D = sup_count.to_dict()
        return C, D
    
    '''
    *Generating candidates*
    in: l<list>[<tuple>], k<int>
    out: <list>[<tuple>]
    '''
    def _apriori_gen(self, l, k): # l:frequent (k-1)-itemsets
        C = set()
        for i in range(len(l)-1):
            for j in range(i+1,len(l)):
                c = tuple(sorted(set(l[i]).union(set(l[j]))))
                if (k == 2) or (len(c) == k) and not (self._has_infrequent_subset(c, l, k-1)):
                    C.add(c)
        return list(C)                
    
    '''
    *Pruning*
    in: c<tuple>, l<list>[<tuple>], k<int>
    out: <boolean>
    '''
    def _has_infrequent_subset(self, c, l, k): # c: candidate k-itemset, L:frequent (k-1)-itemsets
        S = self._ksubsets(set(c), k)
        for s in S:
            if tuple(sorted(s)) not in l:
                return True
        return False
    
    '''
    *Generating all subsets of c with k elements*
    in: c<set>, k<int>
    out: <set> 
    '''
    def _ksubsets(self, c, k):
        import itertools
        return set(itertools.combinations(c, k))
    
    '''
    *Mining all frequent itemsets*
    in: data<pandas.core.frame.DataFrame>
    out: L<list>[<list>[<tuple>]]
    '''
    def mine(self, data):
        import time
        start_time = time.time()
        self.min_sup_count = int(self.min_sup*data.shape[0])+1
        C, D= self._frequent_singleton(data)
        k = 1
        while len(C) > 0:
            C = self._apriori_gen(C, k+1)
            for i in range(data.shape[0]):
                transaction = data.loc[i]
                s = set(transaction[transaction>0].index)
                for c in C:
                    if set(c).issubset(s):
                        D[c] = D.get(c,0)+1
            C = [c for c in C if D.get(c,0) >= self.min_sup_count]
            k += 1
        D = {key:val for key,val in D.items() if val >= self.min_sup_count}
        elapsed_time = time.time() - start_time
        print('Time elapsed:', elapsed_time)
        return D
    
    '''
    *Showing top 10 most frequent 2,3,4-itemsets*
    in: D<dict>
    '''
    def display(self, D):
        for i in range(2,5):
            print('Top 10 most frequent ', i, '-itemset:',sep='')
            L = [item for item in D.items() if len(item[0])==i]
            L = sorted(L, key = lambda item: float(item[1]), reverse = True)
            for j in range(10):
                print(j+1,': ', L[j], sep='')