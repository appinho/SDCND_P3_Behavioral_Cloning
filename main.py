import data
import net


X_train,y_train = data.load_data()
# net.one_layer_net(X_train,y_train,10,'model.h5')
# net.one_layer_net_with_preprocessing(X_train,y_train,7,'model_pre.h5')
net.le_net(X_train,y_train,3,'model_lenet.h5')
