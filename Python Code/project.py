import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from keras.callbacks import EarlyStopping

# reading/standardizing and normalizing the data
data = pd.read_csv("new_data_1.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, 18:19].values

stand_data = StandardScaler().fit_transform(X=data)
norm_data = MinMaxScaler(copy=False).fit_transform(X=stand_data)

# one hot encoding the classes
one_hot = OneHotEncoder()
y = one_hot.fit_transform(y).toarray()

# splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# split the data into training and testing data 5-fold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
res = []

for i, (train, test) in enumerate(cv.split(X_train, y_train)):
    # building the NN
    l1 = keras.regularizers.L1(l1=0.1)
    model = Sequential()
    model.add(Dense(23, input_dim=18, activation='relu', kernel_regularizer=l1))
    model.add(Dense(28, input_dim=23, activation='relu', kernel_regularizer=l1))
    model.add(Dense(5, input_dim=28, activation='softmax', kernel_regularizer=l1))

    # compiling the model
    keras.optimizers.Adam(learning_rate=0.1, ema_momentum=0.6, use_ema=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_squared_error'])

    # defining early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.1, restore_best_weights=True, start_from_epoch=70)

    # training the model
    history = model.fit(X[train], y[train], epochs=500, batch_size=5500, validation_data=(X_test, y_test), callbacks=early_stopping)

    scores = model.evaluate(X[test], y[test], verbose=0)
    res.append(scores)
    print("fold :", i, "accuracy is: ", scores)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss curve')
    plt.ylabel('cross-entropy loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['mean_squared_error'])
    plt.title('accuracy and MSE plot')
    plt.ylabel('accuracy, MSE')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'MSE'], loc='upper left')
    plt.show()

print("Results: ")
for i in range(0, 5):
    print('Fold #', (i+1), ', loss=', round(res[i][0], 5), ', Accuracy=', round(res[i][1], 5), ', MSE=', round(res[i][2], 5))

loss_mean, acc_mean, mse_mean = 0, 0, 0
for i in range(0, 5):
    loss_mean += res[i][0]
    acc_mean += res[i][1]
    mse_mean += res[i][2]

print("Mean values from all 5-folds:")
print("loss=", round(loss_mean/5, 5), ', Accuracy=', round(acc_mean/5, 5), ', MSE=', round(mse_mean/5, 5))

