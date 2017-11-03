from keras.models import Sequential
from keras.layers import Flatten,Dense

# Hyperparameters
ratio_validation_set = 0.3
num_epochs = 10

def one_layer_net(X_train,y_train):
    # define network
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    # training
    train_model(model,X_train,y_train)

def train_model(model,X_train,y_train):
    # define loss and optimizer and train the model
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train,y_train,validation_split=ratio_validation_set,shuffle=True,nb_epoch=num_epochs)

    # save model
    model.save('model.h5')