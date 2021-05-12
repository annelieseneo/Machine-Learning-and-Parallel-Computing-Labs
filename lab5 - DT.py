# module for graphviz's Dot language
import pydotplus

# import tree classifier
from sklearn import tree

# import cross validation
from sklearn import model_selection

# for plotting tree edges
import collections

# load feature variables which are height and length of hair
X=[[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],
   [128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],
   [171,36],[116,25],[196,25]]
print(X)
                                                                           
# load target class labels which are man or woman, binary class
Y = ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman','Man',
     'Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man']
print(Y)

# feature names
data_feature_names = ['height','length of hair']

# 40% test and 60% train
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.40, 
                                                                    random_state = 5)

# load DT classifier
clf = tree.DecisionTreeClassifier()

# train classifier by fitting training data into the model
clf = clf.fit(X_train, Y_train)

# predict test set
preds = clf.predict(X_test)
print('Test data : ' , X_test)
print('Predicted class labels : ', preds)
print('Actual class labels : ', Y_test)

# import accuracy score function / module
from sklearn.metrics import accuracy_score

# compare actual vs predicted to get accuracy score
print([accuracy_score(Y_test, preds)])

# predict new record without checking model performance accuracy
prediction = clf.predict([[133,37]])
print(prediction)
# model predicted 87.5% that this record belongs to a woman

# visualisation using graphviz graphics tool, drawing using dot files
    # nodes are filled colour and rounded box
dot_data = tree.export_graphviz(clf, feature_names=data_feature_names, 
                                out_file = None, filled = True, rounded = True)
print(dot_data)
# fontname, fill colour, labels, gini indexes, number of samples, number of 
    # each label, label distance, label angle, presence of head label

# create pydotplus.graphviz.Dot object
graph = pydotplus.graph_from_dot_data(dot_data)
print(graph)

# edges in a list
edges = collections.defaultdict(list)
print(edges)

# list of edges of nodes and child nodes
for edge in graph.get_edge_list(): 
    edges[edge.get_source()].append(int(edge.get_destination()))
print(edges)

# sort the nodes
for edge in edges: edges[edge].sort()
print(edges)

# create final pydotplus.graphviz.Node object
for i in range(2): dest = graph.get_node(str(edges[edge][i]))[0]
print(dest)

# filled colours for nodes
colors = ('orange', 'yellow')
dest.set_fillcolor(colors[i])

# store in file
graph.write_png('Decisiontree.png') 