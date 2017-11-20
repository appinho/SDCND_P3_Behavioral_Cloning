import data
import net


X_train,y_train = data.load_data()
# data.draw_data(X_train,y_train)
# net.one_layer_net(X_train,y_train,10,'model.h5')
# net.one_layer_net_with_preprocessing(X_train,y_train,7,'model_pre.h5')
# net.le_net(X_train,y_train,5,'model_lenet_s1.h5',0.5)
net.alex_net(X_train,y_train,12,'model_alexnet.h5')
# net.nvidia_net(X_train,y_train,2,'model_nvidia_aug.h5')
