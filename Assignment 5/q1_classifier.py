import random
import pickle as pkl
import argparse
import csv
import numpy as np
import math
import sys
from scipy.stats import chisquare

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take
        
    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''

sys.setrecursionlimit(50000)

# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data


    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
    Xtrain, Ytrain, Xtest = [],[],[]
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int,row[0].split())
            Xtrain.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int,row[0].split())
            Xtest.append(rw)

    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Ytrain.append(rw)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest


num_feats = 274

# computes Entropy given the labels
def entropy(Ytrain):
    total = Ytrain.shape[0]
    pos = len(np.where(Ytrain==1)[0])
    neg = total-pos
    if(pos==0 or neg==0):
        return 0
    else:
        pos_p = pos*1.0/total
        neg_p = neg*1.0/total
        return -1*(pos_p*math.log(pos_p) + neg_p*math.log(neg_p))


# return the best attribute to split on
# inputs -
# Xtrain, Ytrain - Training data at the node
# attr - current attribute
# all_attributes - remaining attributes to be considered 
# curr_entropy - current entropy at this node

def getAttribute(Xtrain, Ytrain, attr, all_attributes, curr_entropy):

    maxx = float('-inf')

    total = Xtrain.shape[0]
    ret_attr = ""

    # for each attribute compute info gain
    # return the attribute with maximum gain

    for att in all_attributes:
        temp = curr_entropy
        distincts, counts = np.unique(Xtrain[:,att].tolist(), return_counts=True)
        for i in range(len(distincts)):
            val = distincts[i]
            indexes = np.where(Xtrain[:,att]==val)
            new_x = Xtrain[indexes[0]]
            new_y = Ytrain[indexes[0]]
            temp -= (counts[i]*1.0/total)*entropy(new_y)
        if(temp>maxx):
            maxx = temp
            ret_attr = att

    return ret_attr

# return the p-value for chi-square statistic
# inputs -
# Xtrain, Ytrain - training data at the node
# attr - current attribute that makes the node
# best_attr - attribute being tested
# pos,neg,total - positive, negative and total training samples at current node

def chsq_p(Xtrain, Ytrain, attr, best_attr, pos, neg, total):

    obs = []
    exp = []

    distincts = np.unique(Xtrain[:,best_attr].tolist())
    for val in distincts:
        indexes = np.where(Xtrain[:,best_attr]==val)
        new_x = Xtrain[indexes[0]]
        new_y = Ytrain[indexes[0]]

        total_v = new_y.shape[0]
        pos_v = len(np.where(new_y==1)[0])
        neg_v = total_v-pos_v

        exp.append((pos*1.0/total)*total_v)
        obs.append(pos_v)

        exp.append((neg*1.0/total)*total_v)
        obs.append(neg_v)

    (ch, p) = chisquare(obs, f_exp=exp)
    return p

# return the root of decision tree generated
# inputs -
# Xtrain, Ytrain - training data at the current node
# attr - current attribute that makes the node
# all_attributes - remaining attributes to be considered
# curr_entropy - current entropy at this node
# pval - threshold p value for chi-square stopping

def id3_with_chsq(Xtrain, Ytrain, attr, all_attributes, curr_entropy, pval):

    # All train labels are 1
    if(Ytrain.all(0)):
        return TreeNode('T')

    # Not even one label is 1
    if(not Ytrain.any(0)):
        return TreeNode('F')

    total = Ytrain.shape[0]
    pos = len(np.where(Ytrain==1)[0])
    neg = total-pos

    if not all_attributes:
        if(pos>=neg):
            return TreeNode('T')  
        else:
            return TreeNode('F')

    best_attr = getAttribute(Xtrain, Ytrain, attr, all_attributes, curr_entropy)

    p_ret = chsq_p(Xtrain, Ytrain, attr, best_attr, pos, neg, total)

    # if pval from ch-sq is more than threshold,
    # look out for next best attribute to split on
    if(p_ret>pval):
        temp = [x for x in all_attributes if x!=best_attr]
        return id3_with_chsq(Xtrain, Ytrain, attr, temp, curr_entropy, pval)

    distincts = np.unique(Xtrain[:,best_attr].tolist())

    root = TreeNode(str(best_attr+1))

    for val in distincts:
        indexes = np.where(Xtrain[:,best_attr]==val)
        new_x = Xtrain[indexes[0]]
        new_y = Ytrain[indexes[0]]
        curr_entropy = entropy(new_y)
        temp = [x for x in all_attributes if x!=best_attr]
        root.nodes[val-1] = id3_with_chsq(new_x, new_y, best_attr, temp, curr_entropy, pval)

    # so that unseen attribute values can be handled by decision tree
    for i in range(5):
        if(root.nodes[i]==-1):
            if(pos>=neg):
                root.nodes[i] = TreeNode('T')
            else:
                root.nodes[i] = TreeNode('F')

    return root

# returns label prediction for given decision tree root and datapoint
def evaluate_datapoint(root,datapoint):
    if root.data == 'T': return 1
    if root.data =='F': return 0
    return evaluate_datapoint(root.nodes[datapoint[int(root.data)-1]-1], datapoint)
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = float(args['p'])
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)
Xtrain = np.matrix(Xtrain)
Ytrain = np.matrix(Ytrain)
Ytrain = np.reshape(Ytrain, (Ytrain.shape[1], 1))


print("Training...")

curr_entropy = entropy(Ytrain)
all_attributes = range(num_feats)
root = id3_with_chsq(Xtrain, Ytrain, None, all_attributes, curr_entropy, pval)

root.save_tree(tree_name)

print("Testing...")

Ypredict = []
#generate random labels
for i in range(0,len(Xtest)):
    Ypredict.append([evaluate_datapoint(root,Xtest[i])])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")


