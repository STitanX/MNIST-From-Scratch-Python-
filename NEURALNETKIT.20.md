
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow

from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Outputs the range of labels
def range_of_vals(x, axis=0):
    return np.max(x, axis=axis) - np.min(x, axis=axis)

label_range = range_of_vals(y_train)

#One-hot encodes the labels\
def one_hot(labels):
    output = np.zeros((len(labels), label_range + 1))
    output[np.arange(len(labels)), labels] = 1
    return output

train_labels = one_hot(y_train)
test_labels = one_hot(y_test)



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, w_l2=0, b_l2=0):
        
        self.weights = -.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = -.1 *np.random.randn(1,n_neurons)
        
        self.w_l2 = w_l2
        self.b_l2 = b_l2
        #self.weights = (np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs))

        #self.biases = np.zeros((1, n_neurons)) + 1

    def forward(self,inputs):
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases
        self.output = output

        return output

    def backward(self, dvalues):
        dweights = np.dot(self.inputs.T, dvalues)  #deadass just chain rule. d/dx sin(x*y) = cos(x*y) * y
        dbiases = np.sum(dvalues, axis=0, keepdims=True) # dont get this though
        # Gradient on values
        output = np.dot(dvalues, self.weights.T)
        
        
        if self.w_l2 > 0:
            dweights += 2 * self.w_l2 * self.weights
            
            
        if self.b_l2 > 0:
            dbiases += 2 * self.b_l2 * self.biases


        return output, dweights, dbiases
    
    def Regularization_Loss(self):
        regularization_loss = 0
        if self.w_l2 > 0:
            regularization_loss += self.w_l2 * np.sum(self.weights * self.weights)
        if self.b_l2 > 0:
            regularization_loss += self.b_l2 * np.sum(self.biases * self.biases)

        return regularization_loss



#Leaky
class Activation_Relu:
    def forward(self,inputs):
        activation_output = np.maximum(0.01*inputs, inputs)
        return activation_output

    def backward(self, dvalues):

        dinputs = dvalues.copy()
        #Zero gradient where input values were negative
        dinputs[dinputs <= 0] = 0.01
        return dinputs



class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        inputs = np.clip(inputs, 1e-4, 1 - 1e-4)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #subtracting all inputs by largest value
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalizing and storing to probabilities
        self.output = probabilities
        return probabilities


class Loss_CategoricalCrossentropy():
  # Forward pass
    def forward(self, y_pred, y_true):
  # Number of samples in a batch

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-4, 1 - 1e-4)

        correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)


    def backward(self, dvalues, y_true): #only works with softmax + categorical loss
        samples = len(dvalues)
        y_true_indices = np.argmax(y_true, axis=1)
        dinputs = dvalues.copy()
        # Calculate gradient
        dinputs[range(samples), y_true_indices] -= 1
        # Normalize gradient
        return dinputs



        
        return regularization_loss
    def calculate(self, y_pred, y_true):
        # Calculate sample losses
        sample_losses = self.forward(y_pred, y_true)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))



    def update_params(self, layer, dW, dB, len_samples):
        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(dW)

                layer.bias_momentums =  np.zeros_like(dB)

            weight_updates = self.momentum * layer.weight_momentums - (self.current_learning_rate * dW)
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - (self.current_learning_rate * dB) 
            layer.bias_momentums = bias_updates

            layer.weights += weight_updates / len_samples
            layer.biases += (bias_updates / len_samples) * 0

        else:
            weight_updates = -self.current_learning_rate * dW
            bias_updates = -self.current_learning_rate * dB

            layer.weights += weight_updates / len_samples
            layer.biases += (bias_updates / len_samples) 

    def post_update_params(self):
        self.iterations += 1


train = x_train
train = train.reshape(len(x_train), 784)
train = train / 255
mean  = np.mean(train)
std = np.std(train)

val = train[59500:60000]
val_labels = train_labels[59500:60000]

train = train[0:50000]
train_labels = train_labels[0:50000]


test = x_test
test = test.reshape(len(x_test), 784)
test= test / 255
outcomes = []
counter = 10


Layer1 = Layer_Dense(784, 64, w_l2=.01, b_l2=0)
Activation1 = Activation_Relu()

Layer2 = Layer_Dense(64, 10, w_l2=0, b_l2=0)
Activation2 = Softmax()

Loss = Loss_CategoricalCrossentropy()

optimizer = Optimizer_SGD(learning_rate=.01,decay = 0.01, momentum=0.95)
#optimizer = Optimizer_RMSprop(learning_rate=.001, decay=0, epsilon=1e-3, rho=0.99)
#optimizer = Optimizer_Adam(learning_rate=0.001, decay=0.001, epsilon=1e-7, beta_1=0.95, beta_2=0.999)
counter=0
accuracies = []
losses = []
lr_rates = []
Leaky_Relus = []
batch=256
z = train
report = 20
val_accuracy = []
for i in range(2000):
    sample=np.random.randint(0,z.shape[0],size=(batch))
    x = train[sample]
    y = train_labels[sample]
    Z1 = Layer1.forward(x)
    A1 = Activation1.forward(Z1)

    Leaky_Relus.append(A1.mean())
    Z2 = Layer2.forward(A1)
    A2 = Activation2.forward(Z2)

    final = Loss.forward(A2, y)
    final += Layer1.Regularization_Loss() + Layer2.Regularization_Loss()
    

    dA2 = Loss.backward(A2, y)
    dZ2, dW2, dB2 = Layer2.backward(dA2)
    dA1 = Activation1.backward(dZ2)
    dZ1, dW1, dB1 = Layer1.backward(dA1)

    
    dB1 = dB1
    dW1 = dW1 
    optimizer.pre_update_params()
    optimizer.update_params(Layer1, dW1, dB1, len(x))

    optimizer.update_params(Layer2, dW2, dB2, len(x))
    optimizer.post_update_params()
    
    
    
    predictions = np.argmax(A2, axis=1)
    class_targets = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==class_targets)
    outcomes.append(accuracy)

    accuracies.append(accuracy)
    losses.append(final.mean())
    lr_rates.append(optimizer.current_learning_rate)
    counter+=1
    
    if i % report == 0:
        #val forward pass
        j = val
        k = val_labels
        Z1 = Layer1.forward(j)
        A1 = Activation1.forward(Z1)
        Z2 = Layer2.forward(A1)
        A2 = Activation2.forward(Z2)
        final = Loss.forward(A2, k)
        
        predictions = np.argmax(A2, axis=1)
        class_targets = np.argmax(k, axis=1)
        v_accuracy = np.mean(predictions==class_targets)
        val_accuracy.append(v_accuracy)
        print("________________")
        print(f"Number {counter}")
        #print(f"A1[0]: {A1[0]}")
        print(f"Mean A1: {A1.mean()}")
        #print(dA1[0])
        #print(Layer1.weights[0])
        print(f"Mean Z1: {Z1.mean()}")
        print(f"Mean B1: {np.mean(Layer1.biases)}")
        #print(dB1)
        print(f"Acc: {np.mean(accuracies[-report:])}. Loss: {np.mean(losses[-report:])}. Val_Acc: {np.mean(val_accuracy[-report:])}. Learning Rate: {np.round(optimizer.current_learning_rate, decimals = 10)}")
        print(f"Val Accuracy: {v_accuracy}")
        print("________________")
print("Done")


test

Z1 = Layer1.forward(test)
A1 = Activation1.forward(Z1)
Z2 = Layer2.forward(A1)
A2 = Activation2.forward(Z2)
final = Loss.forward(A2, test_labels)

test_predictions = np.argmax(A2, axis=1)
class_targets = np.argmax(test_labels, axis=1)
accuracy = np.mean(test_predictions==class_targets)

print("***********************************")
print(f'FINAL TEST ACCURACY OF: {accuracy}')



#Plotting accuracy
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.plot(accuracies)
plt.title('Accuracy over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

#Plotting validation acc
plt.figure(figsize=(10, 5))
plt.subplot(2,2,2)
plt.plot(val_accuracy)
plt.title('Val Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Val Accuracy')
plt.tight_layout()
plt.show()
#Plotting loss
plt.subplot(3, 3, 3)
plt.plot(losses)
plt.title('Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')

#Plotting learning rate
#plt.figure(figsize=(10, 5))
#plt.subplot(2,2,3)
#plt.plot(lr_rates)
#plt.title('Learning Rate over Iterations')
#plt.xlabel('Iteration')
#plt.ylabel('Learning Rate')

#Plotting Relu
plt.figure(figsize=(10, 5))
plt.subplot(3,3,4)
plt.plot(Leaky_Relus)
plt.title('Mean A1 values')
plt.xlabel('Iteration')
plt.ylabel('Mean A1 Values')
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Generate random indices for 25 samples
rand_indices = np.random.randint(0, len(test), 25)

fig, axes = plt.subplots(5, 5, figsize=(14, 14),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.6, wspace=0.6))

# Create a legend for color coding
legend_labels = {'Correct': 'green', 'Incorrect': 'red'}
patchList = []
for key in legend_labels:
    data_key = plt.Rectangle((0, 0), 1, 1, fc=legend_labels[key])
    patchList.append(data_key)
plt.legend(patchList, legend_labels.keys(), loc='upper right')

for i, ax in enumerate(axes.flat):
    idx = rand_indices[i]
    data = test[idx]
    data = data.reshape(28, 28)

    prediction = test_predictions[idx]
    actual = class_targets[idx]

    ax.imshow(data, cmap='gray')

    if prediction == actual:
        color = 'green'
    else:
        color = 'red'

    ax.set_title(f'Pred: {prediction}\nActual: {actual}', color=color, fontsize=16, fontweight='bold')

plt.show()




    
    


