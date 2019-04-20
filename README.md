# Machine-Learning
Developing Basic Machine Learning Models Using Python


The ReLU function is designed mainly to get rid of the vanishing gradient problem in the initial layers of a deep neural network. Since the gradient of ReLU is constant, it suffers less from the vanishing gradient problem. However, since ReLU is linear, it can not map the hidden layers signals to an output (in multiclass or binary classification setting). On the other hand, the output of the Sigmoid and softmax functions are between 0 and 1, (and all the outputs sum up to 1) so we can think of them as probabilities of belonging to each of the classes (binary class in the case of sigmoid and multiple classes in the case of softmax), and therefore very intuitive. Hence we use softmax and sigmoid as the final output layer activations. Please refer to some other materials on multiclass and binary classification as they have not been explained here well
