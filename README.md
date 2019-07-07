# Machine-Learning-with-Tree-Based-Models-in-Python

PART 1 - CLASSIFICATION AND REGRESSION TREES (CART)
- Set of supervised learning models used for problems involving classification and regression

Classification Tree
- Sequence of if-else questions about individual features
- Objective: Infer class labels
- Able to capture non-linear relationships between features and labels
- Don't require feature scaling. For example, it do not need standardization etc.

Decision regions 
- Decision region is the region in the feature space where all the instances are assigned to one class label.
- For example, if result is of two class Pass or Fail. Then there will be 2 decision region. One is Pass region other is Fail region.

Decision boundary
- It is the seperating boundary between two region.
- In above example, decision boundary will be 33% (which is the passing marks)

Logistic regression vs classification tree
- A classification tree divides the feature space into rectangular regions.
- In contrast, a linear model such as logistic regression produces only a single linear decision boundary dividing the feature space into two decision regions.
- In other word, decision boundary produced by logistic regression is linear (straight line) while the boundaries produced by the classification tree divide the feature space into rectangular regions (Not a straight line but boxes/region it divides two class).

Building block of Decision Tree 
- Root: No parent node, question giving rise to two children nodes.
- Internal node: One parent node, question giving rise to two children nodes.
- Lead: One parent node, no children node -> Prediction.

Classication-Tree Learning (Working) - 
- Nodes are grown recursively (based on last node).
- At each node, split the data based on:
1. feature f and split-point(sp) to maximize IG(Information gain from each node).
2. If IG(node)= 0, declare the node a leaf.

Information Gain-
- IG is a synonym for Kullback–Leibler divergence.
- It is the amount of information gained about a random variable or signal from observing another random variable.
- The term is sometimes used synonymously with mutual information, which is the conditional expected value of the Kullback–Leibler divergence.
- KL divergance is the univariate probability distribution of one variable from the conditional distribution of this variable given the other one.

Criteria to measure the impurity of a node I(node):
1. Variance (Regression) [Variance reduction of a node N is defined as the total reduction of the variance of the target variable x due to the split at this node]
2. Gini impurity (Classification) [Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset]
3. Entropy (Classification) [Information entropy is the average rate at which information is produced by a stochastic source of data]




