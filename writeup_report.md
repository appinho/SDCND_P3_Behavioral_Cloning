### Project: Behavioral Cloning

[//]: # (Image References)

[image1]: ./report_images/nvidia_net.png "Neural net architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### 1 Model Architecture and Training Strategy

The chosen neural net architecture follows the model of the paper "End to End Learning for Self-Driving Cars" from NVIDIA (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
Nevertheless, the subsequent layers with their attributes is listed in the following table:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							|
| Cropping2D         		| 66x200x3 RGB image   							|
| Normalization        		| 66x200x3 normalized RGB image   							| 
| Convolution 5x5     	| 1x1 stride, 24 kernels 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same, outputs 31x98x24				|
| Convolution 5x5	    | 1x1 stride, 36 kernels      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same,  outputs 14x47x36 				|
| Convolution 5x5	    | 1x1 stride, 48 kernels      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, padding=same,  outputs 5x22x48				|
| Convolution 3x3	    | 1x1 stride, 64 kernels, padding=valid      									|
| RELU					|	outputs 3x20x64											|
| Convolution 3x3	    | 1x1 stride, 64 kernels, padding=valid      									|
| RELU					|	outputs 1x18x64		
| Flatten       | outputs 1152  |
| Fully connected		| outputs 100        									|
| Fully connected		| outputs 50        									|
| Fully connected		| outputs 10        									|
| Fully connected		| outputs 1        									|

The model first crops the input with an additional Keras Cropping layer to remove parts out of the track (vegetation/sky).
Next, the input is normalized using a Keras lambda layer to have pixel values from -0.5 to 0.5.
Then, 5 Convolution Layers are performed in which each layer includes an RELU activation function to introduce nonlinearity.
Moreover, the first 3 Convolutions are following by a 2x2 stride MaxPool Layer to reduce the feature space.
The end of the network contains 4 fully connected layers to narrow down the information flow to one output neuron which models the resulting steering angle.

#### 2 Attempts to reduce overfitting in the model

The model performs well without further regularization methods.
Nevertheless, dropout layers could be integrated after each pooling layer to generalize the model better.

Finally, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3 Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
Moreover, as loss function the Mean-Squared-Error (MSE) was chosen which is a common loss function for regression models since the steering angle output is a floating number.

#### 4 Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
The model was trained and validated on different types of recorded data to ensure that the model is not overfitting.
Therefore, 2 counter-clockwise loops and 1 clockwise loop of the track were recorded to generalize better in both directions. Additionally, several safety maneuvers that navigate back to the center of the line if the car is at the left or right edge were recorded in both directions. Specifically, on the bridge and in tight curves more recover data was recorded to generalize better on different textures (tiles on bridge) and rare driving cases (like tight curves).

For details about how I created the training data, see the next section. 

### 5 Solution Design Approach

The first step was to use a one-layer fully connected layer to simply setup the whole pipeline.
Therefore, only one recorded driven loop was used to evaluate the first try.
This model had big errors since the input was not normalized.  

The second step was to introduce a normalization layer to the beginning with a Keras lambda layer.
All three image channels were normalized to be bounded between -0.5 and 0.5.
This reduced the losses and reensured that the initialized weights and biases are within the same range as the input.
Nevertheless, the network structure was to simply and couldn't incorporate complex shape searches like edges of the track.  

The third step was to use the LeNet architecture which uses Convolution Layers.
However, the car was not able to drive autonomously because whenever it drifted off the center of the lane it was not able to recover from that offset.
This is why, multiple recovering data was recorded.
As a result, the car could tended to drive leftwards.  

This is due to the fact, that the recordining data was only counter-clockwise. Therefore, the input images were flipped and the recorded steering angles were multiplied by -1 which augments the dataset and removes any tendency of right and left curves.
Moreover, the images from the left and right camera images were used with an correction factor of +-0.2 respectively.
Now, the car could drive up to the bridge but couldn't handle the next narrow curve.  

This is why further data was recorded. First, one more loop in each directions were recorded to have in sum 3 loops of data in both directions to generalize better. Addtionally, multiple recovering manoveurs were recorded on the bridge and in narrow curves to handle these kinds of corner cases.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by a 80/20 ratio respectively. I found that my first tries had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.  

Finally, the LeNet architecture was replaced by the above mentioned NVIDIA end-to-end network structure. This more comprehensive architecture is useful to learn more feature channels and also possess more convolutions layers as well as fully-connected layers to learn more dependencies between the increased input data set. Now, the training and validation losses were both low and within the same region which shows that the overfitting was avoided.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
