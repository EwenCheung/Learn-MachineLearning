import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression

data_cols = [
    "Rented_Bike_Count",
    "Hour",
    "Temperature",
    "Humidity",
    "Wind_speed",
    "Visibility",
    "Dew_point_temperature",
    "Solar_Radiation",
    "Rainfall",
    "Snowfall",
    "Functioning_Day"
]

df = pd.read_csv("../Dataset/SeoulBikeData.csv").drop(["Date", "Holiday", "Seasons"], axis=1)

df.columns = data_cols
df["Functioning_Day"] = (df["Functioning_Day"] == "Yes").astype(int)
df = df[df["Hour"] == 12]  # just loook for the linear regression when hour = 12
df = df.drop(["Hour"], axis=1)

# bike count is target, others is features
# for label in df.columns[1:]:
#     plt.scatter(df[label], df["Rented_Bike_Count"])
#     plt.title(label)
#     plt.ylabel("Bike Count at Noon")
#     plt.xlabel(label)
#     plt.show()

df = df.drop(["Wind_speed", "Visibility", "Functioning_Day"], axis=1)  # drop off where doesnt look linear dont affect.

""" Train Valid Test Split """
train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
    else:
        if len(x_labels) == 1:
            X = dataframe[x_labels[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_labels].values

    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))

    return data, X, y


""" Temperature Regression """
_, X_train_temp, y_train_temp = get_xy(train, "Rented_Bike_Count", x_labels=["Temperature"])
_, X_val_temp, y_val_temp = get_xy(valid, "Rented_Bike_Count", x_labels=["Temperature"])
_, X_test_temp, y_test_temp = get_xy(test, "Rented_Bike_Count", x_labels=["Temperature"])

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
# coefficient and intercept point / R-squared
print(temp_reg.coef_, temp_reg.intercept_)
print(temp_reg.score(X_test_temp, y_test_temp))

""" All Regression """
_, X_train_all, y_train_all = get_xy(train, "Rented_Bike_Count", x_labels=df.columns[1:])
_, X_val_all, y_val_all = get_xy(valid, "Rented_Bike_Count", x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, "Rented_Bike_Count", x_labels=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)
print(all_reg.coef_, all_reg.intercept_)
print(all_reg.score(X_test_all, y_test_all))

""" Regression with Neural Network Temperature """


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()


temp_normalizer = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

temp_nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(1)
])

temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_squared_error")

history = temp_nn_model.fit(
    X_train_temp.reshape(-1), y_train_temp, verbose=0, epochs=100, validation_data=(X_val_temp, y_val_temp)
)

plot_loss(history)

""" Neural Net Temperature """
nn_model = tf.keras.Sequential([
    temp_normalizer,
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

history = nn_model.fit(
    X_train_temp, y_train_temp, verbose=0, epochs=100, validation_data=(X_val_temp, y_val_temp)
)

plot_loss(history)

""" Regression with Neural Network All """


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()


all_normalizer = tf.keras.layers.Normalization(input_shape=(6,), axis=-1)
all_normalizer.adapt(X_train_all)

nn_model = tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history = nn_model.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    verbose=0, epochs=100
)

plot_loss(history)

""" calculate the MSE for both linear reg and nn"""

y_pred_lr = all_reg.predict(X_test_all)
y_pred_nn = nn_model.predict(X_test_all)


def MSE(y_pred, y_real):
    return (np.square(y_pred - y_real)).mean()


print(MSE(y_pred_lr, y_test_all))
print(MSE(y_pred_nn, y_test_all))

""" Prediction """
ax = plt.axes(aspect="equal")
plt.scatter(y_test_all, y_pred_lr, label="Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label="NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims, c="red")
