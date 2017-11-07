from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Convolution2D,MaxPooling2D

# Hyperparameters
ratio_validation_set = 0.3

def one_layer_net(X_train,y_train,num_epochs,model_name):
    # define network
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    # training
    train_model(model,X_train,y_train,num_epochs,model_name)

def one_layer_net_with_preprocessing(X_train,y_train,num_epochs,model_name):
    # define network
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(1))

    # training
    train_model(model,X_train,y_train,num_epochs,model_name)

def le_net(X_train,y_train,num_epochs,model_name):
    input_shape = X_train.shape[1:4]
    print(input_shape)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(6,5,5,activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    print(model.output_shape)
    model.add(Convolution2D(6,5,5,activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D())
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    train_model(model,X_train,y_train,num_epochs,model_name)


def train_model(model,X_train,y_train,num_epochs,model_name):
    # define loss and optimizer and train the model
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train,y_train,validation_split=ratio_validation_set,shuffle=True,nb_epoch=num_epochs)

    # save model
    model.save(model_name)