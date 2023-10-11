import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras import  Sequential

data = []
label = []
os.chdir("./Falls")

cnt_falls = 0
for fileName in os.listdir():
    cnt_falls += 1
    with open(fileName, 'r') as file:
        lines = file.readlines()
    dataTemp = []
    for line in lines:
        elements = [float(x) for x in line.strip().split(',')]
        dataTemp.extend(elements)
    data.append(dataTemp[:230])

fall_labels = np.ones(cnt_falls)

os.chdir("../notFall")

cnt_not_falls = 0
for fileName in os.listdir():
    cnt_not_falls += 1
    with open(fileName, 'r') as file:
        lines = file.readlines()
    dataTemp = []
    for line in lines:
        elements = [float(x) for x in line.strip().split(',')]
        dataTemp.extend(elements)
    data.append(dataTemp[:230])

data = np.array(data)

not_fall_labels = np.zeros(cnt_not_falls)
label = np.concatenate((fall_labels, not_fall_labels))

# print(data.shape)
# print(label.shape)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1)

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=230))  # Assuming input shape is 230

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))  # Assuming it's a binary classification problem

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')