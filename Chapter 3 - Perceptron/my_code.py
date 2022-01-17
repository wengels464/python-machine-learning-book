"""
The purpose of this program is to illustrate how a properly tuned
Perceptron object can perfectly classify the iris dataset.

Pandas is used to create dataframes.

Sklearn provides preprocessing, the Perceptron itself, a tool for splitting
data into test and train, and a score to test the accuracy of the model

Numpy allows us to assign targets in a vectorized manner.

Finally, mlxtend allows us insight into the performance of the Perceptron
over time, as well as the effects of hyperparameter tuning.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from mlxtend.plotting import plot_learning_curves


iris = pd.read_csv('iris.csv')

# We want to add targets for the Perceptron to output
# We code the different species to small integers

conditions = [iris['Species'] == 'Iris-virginica',
              iris['Species'] == 'Iris-versicolor',
              iris['Species'] == 'Iris-setosa']

outputs = [0,1,2]

# np.select is a vectorized form of pandas apply function

targets = np.select(conditions, outputs, "NaN")

# Append the targets

iris['targets'] = targets


# Dropping obvious identifier data

iris = iris.drop(['Id', 'Species'], axis=1)

# Define our total X vector while removing targets

X = iris.drop(['targets'], axis = 1)

# Define our target values

y = iris['targets']

# Break data into 20% test data and 80% train data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)

# Standardize features

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train

pct = Perceptron(max_iter = 25, eta0 = 0.1, random_state=1)
pct.fit(X_train_std, y_train)

# Verify
y_pred = pct.predict(X_test_std)
accuracy_score(y_test, y_pred)

# Visualize (from mlxtend.plotting)
plot_learning_curves(X_train, y_train, X_test, y_test, pct)

print('Accuracy equal to %.3f' % accuracy_score(y_test,y_pred))
