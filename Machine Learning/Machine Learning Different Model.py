import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("../Dataset/magic04.data", names=cols)

df["class"] = (df["class"] == 'g').astype(int)

# for label in cols[:-1]:
#     plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
#     plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
#     plt.title(label)
#     plt.xlabel(label)
#     plt.ylabel("Probability")
#     plt.legend()
# plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler() #make mean = 0 , sd = 1
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler() # make balance data , use if eg len(x) = 100 , len(y) = 1000
        X, y = ros.fit_resample(X, y)

    # stack up two arrays(features and target) together but side by side(horizontal)
    # reshape(y,(len(y),1 (2D array))
    # ... --> X
    # [2.14942082e+00  3.86382163e+00  1.80524080e+00... - 3.10603021e+00
    #  1.00675931e+00  2.90490229e+00]
    # [-7.11651098e-01 - 5.59205901e-01 - 1.16320565e+00...  1.15481083e-01
    #  5.61874684e-01 - 2.66003544e-01]
    # [-8.19992460e-01 - 4.48683623e-01 - 4.08143721e-01... - 3.18668113e-01
    #  1.06193883e+00 - 1.32310207e-01]]
    # [0 0 1... 0 0 1] --> y
    # [[1]
    #  [1]
    #  [1]
    #  ...
    #  [1]
    #  [1]
    #  [0]] --> np.reshape(y, (-1, 1))
    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

""" KNN MODEL """
# from sklearn.neighbors import KNeighborsClassifier
# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train,y_train)
# y_pred = knn_model.predict(X_test)
# print(classification_report(y_test,y_pred))

""" Naive Bayes """
# from sklearn.naive_bayes import GaussianNB
# nb_model = GaussianNB()
# nb_model = nb_model.fit(X_train, y_train)
# y_pred = nb_model.predict(X_test)
# print(classification_report(y_test,y_pred))

""" Log Regression """
# from sklearn.linear_model import LogisticRegression
# lg_model = LogisticRegression()
# lg_model = lg_model.fit(X_train, y_train)
# y_pred = lg_model.predict(X_test)
# print(classification_report(y_test,y_pred))

""" Support Vector Machines (SVM) """
# from sklearn.svm import SVC
# svm_model = SVC()
# svm_model = svm_model.fit(X_train, y_train)
# y_pred = svm_model.predict(X_test)
# print(classification_report(y_test, y_pred))


""" Neural Net (Tensor Flow) """
# import tensorflow as tf
#
#
# def plot_loss(history):
#     plt.plot(history.history["loss"], label="loss")
#     plt.plot(history.history["val_loss"], label="val_loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Binary crossentropy")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def plot_accuracy(history):
#     plt.plot(history.history["accuracy"], label="accuracy")
#     plt.plot(history.history["val_accuracy"], label="val_accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def train_model(X_train, y_train, num_nodes, dropout_prob, learning_rate, batch_size, epochs):
#     nn_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
#         tf.keras.layers.Dropout(dropout_prob),
#         tf.keras.layers.Dense(num_nodes, activation="relu"),
#         tf.keras.layers.Dropout(dropout_prob),
#         tf.keras.layers.Dense(1, activation="sigmoid")
#     ])
#
#     nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
#
#     history = nn_model.fit(
#         X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose = 0
#     )
#
#     return nn_model, history
#
#
# nn_model, history = train_model(X_train, y_train, 32, 0.2, 0.001, 32, 100)
# y_pred = nn_model.predict(X_test)
# y_pred = (y_pred > 0.5).astype(int).reshape(-1, )
# print(classification_report(y_test,y_pred))
