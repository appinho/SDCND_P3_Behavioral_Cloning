import data
import net

X_train,y_train = data.load_data()
net.one_layer_net(X_train,y_train)