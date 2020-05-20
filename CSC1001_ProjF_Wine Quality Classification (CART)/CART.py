'''Tree Node'''
class Node:
    def __init__(self, gini, size, label):
        self.gini = gini        # gini index at the node
        self.size = size        # size of the trainning data
        self.label = label      # class label of the majority at the node
        self.leftchild = None   # left child node
        self.rightchild = None  # right child node
        self.index = None       # index of the attribute for splitting
        self.threshold = None   # splitting criterion

'''Decision Tree'''
class DecisionTree:
    # initializer
    def __init__(self,max_depth=100):
        self.max_depth = max_depth
        self.tree = None

    # select the best attribute for splitting
    def split(self, X, y, init_gini, parent_class):
        n = len(y)
        # if there is only one record or all the records has the same label
        if (n <= 1) or (init_gini == 0):
            return None, None
        
        gini = init_gini
        index = None
        threshold = None

        for idx in range(len(self.attributes)):
            col = [row[idx] for row in X] # column slices
            attributes, classes = zip(*sorted(zip(col, y))) # sort the data by attribute column 
            left_class = dict.fromkeys(parent_class.keys(), 0) # number of each class in the left node
            right_class = parent_class.copy() # number of each class in the right node
            for i in range(1, n):
                k = classes[i-1]
                left_class[k] += 1
                right_class[k] -= 1
                left_gini = 1 - sum([(left_class[key] / i) ** 2 for key in left_class.keys()])
                right_gini = 1 - sum([(right_class[key] / (n-i)) ** 2 for key in right_class.keys()])
                tmp_gini = (i * left_gini + (n-i) * right_gini) / n

                if attributes[i] == attributes[i - 1]:
                    continue

                if tmp_gini < gini: # if information gain > 0, i.e., gini < gini_init 
                    gini = tmp_gini
                    index = idx
                    threshold = (attributes[i]+ attributes[i-1]) / 2

        return index, threshold # return the splitting criterion

    # build the decision tree by recursion
    def build_tree(self, X, y, depth):

        n = len(y)
        parent_class = dict()
        for i in range(n):
            parent_class[y[i]] = parent_class.get(y[i], 0) + 1
        gini = 1 - sum([(parent_class[key]/n)**2 for key in parent_class.keys()])
        class_label = max(parent_class, key = parent_class.get)

        node = Node(gini = gini, size = len(y), label = class_label) # parent node

        if  depth < self.max_depth: # keep splitting
            node.index, node.threshold = self.split(X, y, gini, parent_class)
            if node.index: # if index is not None
                X_left = list()
                y_left = list()
                X_right = list()
                y_right = list()
                for i in range(n):
                    if X[i][node.index] < node.threshold:
                        X_left.append(X[i])
                        y_left.append(y[i])
                    else:
                        X_right.append(X[i])
                        y_right.append(y[i])

                # generate left and right child recursively
                node.leftchild = self.build_tree(X_left, y_left, depth+1)
                node.rightchild = self.build_tree(X_right, y_right, depth+1)

        return node

    # fit the train data to build the decison tree
    def fit(self, headers, train_X, train_y):  
        self.attributes = headers[:-1]
        self.tree = self.build_tree(train_X, train_y, 0)

    # follow the splitting criterion at each node to get the class label at leaf node
    def predict(self, test_X):
        pred_y = list()
        for row in test_X:
            node = self.tree
            while node.index:
                if row[node.index] < node.threshold:
                    node = node.leftchild
                else:
                    node = node.rightchild
            pred_y.append(node.label)
        return pred_y

'''Data Loader'''
def load_data(file_name):
    import csv 
    X = list() # predictors
    y = list() # response 
    with open(file_name) as f:
        reader = csv.reader(f)
        headers = list(map(str.strip, next(reader))) # headers of the data
        for i in reader:
            row = list(map(float, i)) # convert rows to floating number 
            X.append(row[:-1])
            y.append((row[-1] > 6) + 0) # 1 if quatility score > 6 and 0 otherwise
    return headers, X, y

'''Evaluator'''
def evaluate(pred, real):
    cm = [[0,0],[0,0]]
    for i in range(len(pred)):
        if (pred[i] == 0):
            if (real[i] == 0):
                cm[0][0] += 1
            else:
                cm[0][1] += 1
        else:
            if (real[i] == 0):
                cm[1][0] += 1
            else:
                cm[1][1] += 1
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1])
    print("Confusion Matrix:")
    print("\tPredicted")
    print("True\tNo\tYes")
    print("No\t" + str(cm[0][0]) + "\t" + str(cm[0][1]))
    print("Yes\t" + str(cm[1][0]) + "\t" + str(cm[1][1]))
    print("Accuracy: " + str(accuracy))

'''Main'''
def main():
    headers, train_X, train_y = load_data("train.csv")
    

    print("\nHello, I'm the decision tree classifier!")
    clf = DecisionTree(7) # set max_depth = 7 to avoid overfitting
    print("Training data: " + str(len(train_y)) + " x (" + str(len(train_X[0])) + "+1)\n")
    clf.fit(headers,train_X,train_y)

    headers, test_X, test_y = load_data("test.csv")
    print("Test data: " + str(len(test_y)) + " x (" + str(len(test_X[0])) + "+1)")
    pred_y = clf.predict(test_X)
    evaluate(pred_y, test_y)

    print("\nBye, my decision tree classifier...\n")
 
main()