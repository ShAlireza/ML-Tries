import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from pydotplus import graph_from_dot_data

from utils import plot_decision_regions

from O3 import prepare_data

X_train, X_test, y_train, y_test = prepare_data(standardize=True,
                                                split=True)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=4,
                                       random_state=1)

decision_tree.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=decision_tree,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Or just visualize tree using scikit-learn tree module.

tree.plot_tree(decision_tree)
plt.show()

# Or using graphviz for visualizing out tree.

# dot_data = export_graphviz(decision_tree, filled=True,
#                            rounded=True, class_names=['Setosa', 'Versicolor',
#                                                       'Virginica'],
#                            feature_names=['petal length', 'petal width'],
#                            out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('tree.png')

print(f'Accuracy: {decision_tree.score(X_test, y_test) * 100}%')
