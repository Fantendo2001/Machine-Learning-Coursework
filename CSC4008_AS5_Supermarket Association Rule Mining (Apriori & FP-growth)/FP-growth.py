class Node:
    def __init__(self, item, sup):
        self.item = item
        self.sup = sup
        self.adjacent = None
        self.tail = None

class treeNode:
    def __init__(self, item, count=1):
        self.item = item
        self.count = count
        self.parent = None
        self.next = None
        self.child = set()
    
    def single_path(self):
        temp = self
        while len(temp.child)>0:
            if len(temp.child)>1:
                return False
            temp = tuple(temp.child)[0]
        return True

    def insert_tree(self, header, transaction, i, count = 1):
        if i >= len(transaction):
            return
        head = None
        for node in header:
            if node.item == transaction[i]:
                head = node
                break
        if head == None:
            return
        for c in self.child:
            if c.item == transaction[i]:
                c.count += count
                c.insert_tree(header,transaction, i+1, count)
                return
        c = self.__class__(transaction[i],count)
        c.parent = self
        self.child.add(c)
        if head.adjacent == None:
            head.tail = head.adjacent = c
        else:
            head.tail.next = c
            head.tail = head.tail.next
        c.insert_tree(header,transaction, i+1, count)
        return

class FP_tree:
    def __init__(self,header, tree):
        self.header = header
        self.tree = tree
        
    '''
    *Generating conditional pattern base*
    in: item<str>
    out: pattern_base<dict>
    '''
    def _cond_pattern_base(self, item):
        pattern_base = dict()
        for node in self.header:
            if node.item == item:
                break
        head = node.adjacent
        while True:
            temp = head
            l = list()
            while temp.parent.item != '':
                l.append(temp.parent.item)
                temp = temp.parent
            if len(l) > 0:
                l.reverse()
                pattern_base[tuple(l)] = head.count
            if head.next == None:
                break
            head = head.next
        return pattern_base
        
    '''
    *Generating conditional FP-tree*
    in: item<str>, min_sup_count<int>
    out: <FP-tree>
    '''
    def cond_FP_tree(self, item, min_sup_count):
        cpb = self._cond_pattern_base(item)
        root = treeNode('',0)
        header = dict()
        for itemset in cpb:
            for item in itemset:
                header[item] = header.get(item,0) + cpb[itemset]
        header = [Node(key,val) for key,val in header.items() if val >= min_sup_count]        
        for itemset in cpb:
            root.insert_tree(header, itemset, 0, cpb[itemset])
        return FP_tree(header, root)

class FP_growth:
    def __init__(self, min_sup = 0.2):
        self.min_sup = min_sup
    
    '''
    *Generating frequent 1-itemsets*
    in: data<pandas.core.frame.DataFrame>
    out: header<list>[<Node>], D<dict>
    '''
    def _frequent_item(self, data):
        sup_count = data.sum()
        sup_count = sup_count[sup_count >= self.min_sup_count]
        sup_count = sup_count.sort_values(ascending=False)
        header = [Node(item,sup_count[item]) for item in sup_count.index]
        sup_count.index = [(idx,) for idx in sup_count.index]
        D = sup_count.to_dict()
        return header, D
    
    '''
    *Constructing tree*
    in: header<list>[<Node>], data<pandas.core.frame.DataFrame>
    out: root<treeNode>
    '''        
    def _construct_tree(self, header, data):
        root = treeNode('',0)
        for i in range(data.shape[0]):
            transaction = data.loc[i] 
            transaction = [node.item for node in header if transaction[node.item] > 0]
            root.insert_tree(header, transaction, 0)
        return root
    
    '''
    *Implementing FP-growth*
    in: fp_tree<FP-tree>, pattern<list>
    out: D<dict>
    '''    
    def _FP_growth(self, fp_tree, pattern):
        import itertools
        D = dict()
        if fp_tree.tree.single_path():
            temp = fp_tree.tree
            l = list()
            d = dict()
            while len(temp.child) > 0:
                c = tuple(temp.child)[0]
                l.append(c.item)
                d[c.item] = c.count
                temp = c
            for i in range(0,len(l)):
                combination = list(itertools.combinations(l,i+1))
                for itemset in combination:
                    sup_count = d[itemset[-1]]
                    itemset = list(itemset)+ pattern
                    D[tuple(itemset)] = sup_count
        else:
            for node in fp_tree.header:
                cfpt = fp_tree.cond_FP_tree(node.item, self.min_sup_count)
                D[tuple([node.item]+pattern)] = node.sup
                if len(cfpt.tree.child) > 0:
                    D = {**D,**self._FP_growth(cfpt, [node.item]+pattern)}
        return D 
        
    '''
    *Mining all frequent itemsets*
    in: data<pandas.core.frame.DataFrame>
    out: <dict>
    '''
    def mine(self,data):
        import time
        start_time = time.time()
        self.min_sup_count = int(self.min_sup*data.shape[0])+1
        header, self.D = self._frequent_item(data) # D may be unnecessary
        tree = self._construct_tree(header,data)
        fp_tree = FP_tree(header, tree)
        self.D = {**self.D,**self._FP_growth(fp_tree,[])}
        elapsed_time = time.time() - start_time
        print('Time elapsed:', elapsed_time)
        return self.D
    
    '''
    *Showing top 10 most frequent 2,3,4-itemsets*
    in: D<dict>
    '''
    def display(self,D):
        for i in range(2,5):
            print('Top 10 most frequent ', i, '-itemset:',sep='')
            L = [(tuple(sorted(item[0])),item[1]) for item in D.items() if len(item[0])==i]
            L = sorted(L, key = lambda item: float(item[1]), reverse = True)
            for j in range(10):
                print(j+1,': ', L[j], sep='')