```python
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



test_labels.shape

```




    (10000, 10)




```python
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

```


```python
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




```


```python
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


```


```python
val_labels.shape
```




    (500, 10)




```python
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

```

    ________________
    Number 1
    Mean A1: 0.33412939039741596
    Mean Z1: -0.07194750430543591
    Mean B1: -0.009016341315744657
    Acc: 0.15234375. Loss: 7.279438137568702. Val_Acc: 0.186. Learning Rate: 0.01
    Val Accuracy: 0.186
    ________________
    ________________
    Number 21
    Mean A1: -0.01943553605804145
    Mean Z1: -2.4017127365528395
    Mean B1: -0.009016341315744657
    Acc: 0.1869140625. Loss: 7.291675376828389. Val_Acc: 0.159. Learning Rate: 0.0083333333
    Val Accuracy: 0.132
    ________________
    ________________
    Number 41
    Mean A1: -0.06369868413782472
    Mean Z1: -6.369868413782471
    Mean B1: -0.009016341315744657
    Acc: 0.10390625. Loss: 7.683269934984983. Val_Acc: 0.14266666666666666. Learning Rate: 0.0071428571
    Val Accuracy: 0.11
    ________________
    ________________
    Number 61
    Mean A1: -0.10552014106925948
    Mean Z1: -10.552014106925949
    Mean B1: -0.009016341315744657
    Acc: 0.09453125. Loss: 8.726109109283572. Val_Acc: 0.1345. Learning Rate: 0.00625
    Val Accuracy: 0.11
    ________________
    ________________
    Number 81
    Mean A1: -0.14507557582239847
    Mean Z1: -14.507557582239846
    Mean B1: -0.009016341315744657
    Acc: 0.12734375. Loss: 10.422830996499998. Val_Acc: 0.138. Learning Rate: 0.0055555556
    Val Accuracy: 0.152
    ________________
    ________________
    Number 101
    Mean A1: -0.1814513110876283
    Mean Z1: -18.14513110876283
    Mean B1: -0.009016341315744657
    Acc: 0.223046875. Loss: 12.602414555742765. Val_Acc: 0.155. Learning Rate: 0.005
    Val Accuracy: 0.24
    ________________
    ________________
    Number 121
    Mean A1: -0.21492587053729126
    Mean Z1: -21.492587053729128
    Mean B1: -0.009016341315744657
    Acc: 0.234765625. Loss: 15.118281547833067. Val_Acc: 0.16142857142857145. Learning Rate: 0.0045454545
    Val Accuracy: 0.2
    ________________
    ________________
    Number 141
    Mean A1: -0.24574479415644582
    Mean Z1: -24.57447941564458
    Mean B1: -0.009016341315744657
    Acc: 0.2263671875. Loss: 17.867358006145547. Val_Acc: 0.16625. Learning Rate: 0.0041666667
    Val Accuracy: 0.2
    ________________
    ________________
    Number 161
    Mean A1: -0.2742607024378207
    Mean Z1: -27.426070243782064
    Mean B1: -0.009016341315744657
    Acc: 0.2341796875. Loss: 20.78327294817534. Val_Acc: 0.1748888888888889. Learning Rate: 0.0038461538
    Val Accuracy: 0.244
    ________________
    ________________
    Number 181
    Mean A1: -0.3008447128341664
    Mean Z1: -30.084471283416644
    Mean B1: -0.009016341315744657
    Acc: 0.2783203125. Loss: 23.8079891238281. Val_Acc: 0.18480000000000002. Learning Rate: 0.0035714286
    Val Accuracy: 0.274
    ________________
    ________________
    Number 201
    Mean A1: -0.3258220667696529
    Mean Z1: -32.582206676965285
    Mean B1: -0.009016341315744657
    Acc: 0.303125. Loss: 26.924272413056833. Val_Acc: 0.1969090909090909. Learning Rate: 0.0033333333
    Val Accuracy: 0.318
    ________________
    ________________
    Number 221
    Mean A1: -0.34928880782106864
    Mean Z1: -34.92888078210686
    Mean B1: -0.009016341315744657
    Acc: 0.3830078125. Loss: 30.097638242131165. Val_Acc: 0.21333333333333335. Learning Rate: 0.003125
    Val Accuracy: 0.394
    ________________
    ________________
    Number 241
    Mean A1: -0.3715401390868101
    Mean Z1: -37.15401390868101
    Mean B1: -0.009016341315744657
    Acc: 0.3564453125. Loss: 33.3190669370116. Val_Acc: 0.22338461538461538. Learning Rate: 0.0029411765
    Val Accuracy: 0.344
    ________________
    ________________
    Number 261
    Mean A1: -0.3926292754045785
    Mean Z1: -39.26292754045785
    Mean B1: -0.009016341315744657
    Acc: 0.3533203125. Loss: 36.564948282851915. Val_Acc: 0.23314285714285712. Learning Rate: 0.0027777778
    Val Accuracy: 0.36
    ________________
    ________________
    Number 281
    Mean A1: -0.41278176825943946
    Mean Z1: -41.278176825943945
    Mean B1: -0.009016341315744657
    Acc: 0.3390625. Loss: 39.83420943952021. Val_Acc: 0.24159999999999998. Learning Rate: 0.0026315789
    Val Accuracy: 0.36
    ________________
    ________________
    Number 301
    Mean A1: -0.43207242423897285
    Mean Z1: -43.20724242389729
    Mean B1: -0.009016341315744657
    Acc: 0.3703125. Loss: 43.12875720400368. Val_Acc: 0.25275000000000003. Learning Rate: 0.0025
    Val Accuracy: 0.42
    ________________
    ________________
    Number 321
    Mean A1: -0.45065677638454427
    Mean Z1: -45.06567763845443
    Mean B1: -0.009016341315744657
    Acc: 0.3953125. Loss: 46.43511707371469. Val_Acc: 0.2623529411764706. Learning Rate: 0.0023809524
    Val Accuracy: 0.416
    ________________
    ________________
    Number 341
    Mean A1: -0.4686487310986281
    Mean Z1: -46.864873109862806
    Mean B1: -0.009016341315744657
    Acc: 0.4103515625. Loss: 49.77878161478352. Val_Acc: 0.2744444444444445. Learning Rate: 0.0022727273
    Val Accuracy: 0.48
    ________________
    ________________
    Number 361
    Mean A1: -0.4859427469519152
    Mean Z1: -48.594274695191515
    Mean B1: -0.009016341315744657
    Acc: 0.4419921875. Loss: 53.14355225872102. Val_Acc: 0.2862105263157895. Learning Rate: 0.002173913
    Val Accuracy: 0.498
    ________________
    ________________
    Number 381
    Mean A1: -0.5026629999323424
    Mean Z1: -50.26629999323424
    Mean B1: -0.009016341315744657
    Acc: 0.468359375. Loss: 56.498504614460536. Val_Acc: 0.29930000000000007. Learning Rate: 0.0020833333
    Val Accuracy: 0.548
    ________________
    ________________
    Number 401
    Mean A1: -0.5187524763533724
    Mean Z1: -51.875247635337246
    Mean B1: -0.009016341315744657
    Acc: 0.4767578125. Loss: 59.85382433450134. Val_Acc: 0.3155. Learning Rate: 0.002
    Val Accuracy: 0.51
    ________________
    ________________
    Number 421
    Mean A1: -0.53440510287139
    Mean Z1: -53.44051028713899
    Mean B1: -0.009016341315744657
    Acc: 0.458203125. Loss: 63.205313309450844. Val_Acc: 0.3353. Learning Rate: 0.0019230769
    Val Accuracy: 0.528
    ________________
    ________________
    Number 441
    Mean A1: -0.5496501813797181
    Mean Z1: -54.96501813797181
    Mean B1: -0.009016341315744657
    Acc: 0.4818359375. Loss: 66.56541346854657. Val_Acc: 0.3569. Learning Rate: 0.0018518519
    Val Accuracy: 0.542
    ________________
    ________________
    Number 461
    Mean A1: -0.5644532033600551
    Mean Z1: -56.44532033600551
    Mean B1: -0.009016341315744657
    Acc: 0.483984375. Loss: 69.94035977841402. Val_Acc: 0.3779. Learning Rate: 0.0017857143
    Val Accuracy: 0.53
    ________________
    ________________
    Number 481
    Mean A1: -0.5788755617271226
    Mean Z1: -57.88755617271225
    Mean B1: -0.009016341315744657
    Acc: 0.453125. Loss: 73.3142507728227. Val_Acc: 0.3952. Learning Rate: 0.0017241379
    Val Accuracy: 0.498
    ________________
    ________________
    Number 501
    Mean A1: -0.5929297414385404
    Mean Z1: -59.29297414385404
    Mean B1: -0.009016341315744657
    Acc: 0.4849609375. Loss: 76.68137560299682. Val_Acc: 0.41150000000000003. Learning Rate: 0.0016666667
    Val Accuracy: 0.566
    ________________
    ________________
    Number 521
    Mean A1: -0.6066496318186414
    Mean Z1: -60.66496318186414
    Mean B1: -0.009016341315744657
    Acc: 0.5166015625. Loss: 80.050730466712. Val_Acc: 0.4314000000000001. Learning Rate: 0.0016129032
    Val Accuracy: 0.598
    ________________
    ________________
    Number 541
    Mean A1: -0.620036315140461
    Mean Z1: -62.00363151404609
    Mean B1: -0.009016341315744657
    Acc: 0.4927734375. Loss: 83.40953838676676. Val_Acc: 0.44650000000000006. Learning Rate: 0.0015625
    Val Accuracy: 0.502
    ________________
    ________________
    Number 561
    Mean A1: -0.6331377173398464
    Mean Z1: -63.313771733984645
    Mean B1: -0.009016341315744657
    Acc: 0.46953125. Loss: 86.77450820040363. Val_Acc: 0.4593. Learning Rate: 0.0015151515
    Val Accuracy: 0.5
    ________________
    ________________
    Number 581
    Mean A1: -0.6459971161114951
    Mean Z1: -64.5997116111495
    Mean B1: -0.009016341315744657
    Acc: 0.4853515625. Loss: 90.14397913518133. Val_Acc: 0.4739. Learning Rate: 0.0014705882
    Val Accuracy: 0.566
    ________________
    ________________
    Number 601
    Mean A1: -0.6586178373389042
    Mean Z1: -65.86178373389042
    Mean B1: -0.009016341315744657
    Acc: 0.5115234375. Loss: 93.50523268666907. Val_Acc: 0.4870000000000001. Learning Rate: 0.0014285714
    Val Accuracy: 0.58
    ________________
    ________________
    Number 621
    Mean A1: -0.6710084753472414
    Mean Z1: -67.10084753472415
    Mean B1: -0.009016341315744657
    Acc: 0.4984375. Loss: 96.8782137473347. Val_Acc: 0.4944. Learning Rate: 0.0013888889
    Val Accuracy: 0.542
    ________________
    ________________
    Number 641
    Mean A1: -0.683107061201609
    Mean Z1: -68.31070612016089
    Mean B1: -0.009016341315744657
    Acc: 0.4876953125. Loss: 100.25918888406459. Val_Acc: 0.5025999999999999. Learning Rate: 0.0013513514
    Val Accuracy: 0.508
    ________________
    ________________
    Number 661
    Mean A1: -0.6949246959253549
    Mean Z1: -69.49246959253549
    Mean B1: -0.009016341315744657
    Acc: 0.4912109375. Loss: 103.60778068385954. Val_Acc: 0.5126000000000001. Learning Rate: 0.0013157895
    Val Accuracy: 0.56
    ________________
    ________________
    Number 681
    Mean A1: -0.7064975710374274
    Mean Z1: -70.64975710374274
    Mean B1: -0.009016341315744657
    Acc: 0.51328125. Loss: 106.95807729487596. Val_Acc: 0.5263000000000001. Learning Rate: 0.0012820513
    Val Accuracy: 0.634
    ________________
    ________________
    Number 701
    Mean A1: -0.7178370933982798
    Mean Z1: -71.783709339828
    Mean B1: -0.009016341315744657
    Acc: 0.575. Loss: 110.28418456204156. Val_Acc: 0.5395. Learning Rate: 0.00125
    Val Accuracy: 0.684
    ________________
    ________________
    Number 721
    Mean A1: -0.7290081965297693
    Mean Z1: -72.90081965297694
    Mean B1: -0.009016341315744657
    Acc: 0.5568359375. Loss: 113.6087808938781. Val_Acc: 0.5487. Learning Rate: 0.0012195122
    Val Accuracy: 0.6
    ________________
    ________________
    Number 741
    Mean A1: -0.7400328312911073
    Mean Z1: -74.00328312911073
    Mean B1: -0.009016341315744657
    Acc: 0.5318359375. Loss: 116.93921383788333. Val_Acc: 0.554. Learning Rate: 0.0011904762
    Val Accuracy: 0.586
    ________________
    ________________
    Number 761
    Mean A1: -0.7508769307939948
    Mean Z1: -75.08769307939947
    Mean B1: -0.009016341315744657
    Acc: 0.523046875. Loss: 120.27993871591532. Val_Acc: 0.5582999999999999. Learning Rate: 0.0011627907
    Val Accuracy: 0.584
    ________________
    ________________
    Number 781
    Mean A1: -0.761537647288452
    Mean Z1: -76.15376472884519
    Mean B1: -0.009016341315744657
    Acc: 0.5228515625. Loss: 123.60831183993405. Val_Acc: 0.5591999999999999. Learning Rate: 0.0011363636
    Val Accuracy: 0.566
    ________________
    ________________
    Number 801
    Mean A1: -0.7720359087787851
    Mean Z1: -77.20359087787851
    Mean B1: -0.009016341315744657
    Acc: 0.487890625. Loss: 126.93673572553175. Val_Acc: 0.5611. Learning Rate: 0.0011111111
    Val Accuracy: 0.548
    ________________
    ________________
    Number 821
    Mean A1: -0.7823758278126899
    Mean Z1: -78.237582781269
    Mean B1: -0.009016341315744657
    Acc: 0.4802734375. Loss: 130.2633632350584. Val_Acc: 0.5605. Learning Rate: 0.0010869565
    Val Accuracy: 0.516
    ________________
    ________________
    Number 841
    Mean A1: -0.7925876491813654
    Mean Z1: -79.25876491813652
    Mean B1: -0.009016341315744657
    Acc: 0.494140625. Loss: 133.5839830551809. Val_Acc: 0.5602. Learning Rate: 0.0010638298
    Val Accuracy: 0.536
    ________________
    ________________
    Number 861
    Mean A1: -0.802615344618763
    Mean Z1: -80.2615344618763
    Mean B1: -0.009016341315744657
    Acc: 0.5123046875. Loss: 136.90368289883727. Val_Acc: 0.5646. Learning Rate: 0.0010416667
    Val Accuracy: 0.618
    ________________
    ________________
    Number 881
    Mean A1: -0.8125181127489911
    Mean Z1: -81.25181127489911
    Mean B1: -0.009016341315744657
    Acc: 0.5447265625. Loss: 140.20799303795224. Val_Acc: 0.5713. Learning Rate: 0.0010204082
    Val Accuracy: 0.632
    ________________
    ________________
    Number 901
    Mean A1: -0.822285501763177
    Mean Z1: -82.2285501763177
    Mean B1: -0.009016341315744657
    Acc: 0.5359375. Loss: 143.52770667618012. Val_Acc: 0.5747. Learning Rate: 0.001
    Val Accuracy: 0.634
    ________________
    ________________
    Number 921
    Mean A1: -0.8318910702452098
    Mean Z1: -83.18910702452096
    Mean B1: -0.009016341315744657
    Acc: 0.546484375. Loss: 146.83262502603202. Val_Acc: 0.5751000000000001. Learning Rate: 0.0009803922
    Val Accuracy: 0.606
    ________________
    ________________
    Number 941
    Mean A1: -0.8413958392417239
    Mean Z1: -84.1395839241724
    Mean B1: -0.009016341315744657
    Acc: 0.527734375. Loss: 150.1343558885619. Val_Acc: 0.581. Learning Rate: 0.0009615385
    Val Accuracy: 0.62
    ________________
    ________________
    Number 961
    Mean A1: -0.8507580695684814
    Mean Z1: -85.07580695684814
    Mean B1: -0.009016341315744657
    Acc: 0.532421875. Loss: 153.42281655512215. Val_Acc: 0.5862999999999999. Learning Rate: 0.0009433962
    Val Accuracy: 0.606
    ________________
    ________________
    Number 981
    Mean A1: -0.8599998631797094
    Mean Z1: -85.99998631797095
    Mean B1: -0.009016341315744657
    Acc: 0.53203125. Loss: 156.71414033030172. Val_Acc: 0.5884999999999999. Learning Rate: 0.0009259259
    Val Accuracy: 0.61
    ________________
    ________________
    Number 1001
    Mean A1: -0.8691376174945438
    Mean Z1: -86.91376174945437
    Mean B1: -0.009016341315744657
    Acc: 0.526171875. Loss: 160.00128353485087. Val_Acc: 0.5872999999999999. Learning Rate: 0.0009090909
    Val Accuracy: 0.556
    ________________
    ________________
    Number 1021
    Mean A1: -0.8781428794843318
    Mean Z1: -87.81428794843318
    Mean B1: -0.009016341315744657
    Acc: 0.500390625. Loss: 163.2813425998443. Val_Acc: 0.5862999999999998. Learning Rate: 0.0008928571
    Val Accuracy: 0.522
    ________________
    ________________
    Number 1041
    Mean A1: -0.8870610499806961
    Mean Z1: -88.70610499806962
    Mean B1: -0.009016341315744657
    Acc: 0.486328125. Loss: 166.57412251054856. Val_Acc: 0.5910999999999998. Learning Rate: 0.000877193
    Val Accuracy: 0.604
    ________________
    ________________
    Number 1061
    Mean A1: -0.8959032828598612
    Mean Z1: -89.59032828598609
    Mean B1: -0.009016341315744657
    Acc: 0.5404296875. Loss: 169.84090525001858. Val_Acc: 0.594. Learning Rate: 0.000862069
    Val Accuracy: 0.618
    ________________
    ________________
    Number 1081
    Mean A1: -0.9046350621803826
    Mean Z1: -90.46350621803823
    Mean B1: -0.009016341315744657
    Acc: 0.534375. Loss: 173.1308520460845. Val_Acc: 0.5929. Learning Rate: 0.0008474576
    Val Accuracy: 0.612
    ________________
    ________________
    Number 1101
    Mean A1: -0.9132663177161716
    Mean Z1: -91.32663177161716
    Mean B1: -0.009016341315744657
    Acc: 0.5337890625. Loss: 176.40905345368103. Val_Acc: 0.5888. Learning Rate: 0.0008333333
    Val Accuracy: 0.602
    ________________
    ________________
    Number 1121
    Mean A1: -0.9217816050984643
    Mean Z1: -92.17816050984644
    Mean B1: -0.009016341315744657
    Acc: 0.5236328125. Loss: 179.66981141708484. Val_Acc: 0.5893. Learning Rate: 0.0008196721
    Val Accuracy: 0.61
    ________________
    ________________
    Number 1141
    Mean A1: -0.9302104584170062
    Mean Z1: -93.02104584170061
    Mean B1: -0.009016341315744657
    Acc: 0.5248046875. Loss: 182.93311482622403. Val_Acc: 0.5892000000000001. Learning Rate: 0.0008064516
    Val Accuracy: 0.584
    ________________
    ________________
    Number 1161
    Mean A1: -0.9385854038959215
    Mean Z1: -93.85854038959215
    Mean B1: -0.009016341315744657
    Acc: 0.509375. Loss: 186.20644579207934. Val_Acc: 0.5879. Learning Rate: 0.0007936508
    Val Accuracy: 0.558
    ________________
    ________________
    Number 1181
    Mean A1: -0.9468700484212548
    Mean Z1: -94.68700484212546
    Mean B1: -0.009016341315744657
    Acc: 0.508984375. Loss: 189.47697716300863. Val_Acc: 0.5883. Learning Rate: 0.00078125
    Val Accuracy: 0.574
    ________________
    ________________
    Number 1201
    Mean A1: -0.9551009036003242
    Mean Z1: -95.51009036003242
    Mean B1: -0.009016341315744657
    Acc: 0.537890625. Loss: 192.73290977776196. Val_Acc: 0.59. Learning Rate: 0.0007692308
    Val Accuracy: 0.582
    ________________
    ________________
    Number 1221
    Mean A1: -0.9632391928660917
    Mean Z1: -96.32391928660917
    Mean B1: -0.009016341315744657
    Acc: 0.5306640625. Loss: 196.0215230439582. Val_Acc: 0.5937. Learning Rate: 0.0007575758
    Val Accuracy: 0.59
    ________________
    ________________
    Number 1241
    Mean A1: -0.9712680827282432
    Mean Z1: -97.1268082728243
    Mean B1: -0.009016341315744657
    Acc: 0.51640625. Loss: 199.29608793733203. Val_Acc: 0.5962999999999999. Learning Rate: 0.0007462687
    Val Accuracy: 0.588
    ________________
    ________________
    Number 1261
    Mean A1: -0.9791826222427044
    Mean Z1: -97.91826222427044
    Mean B1: -0.009016341315744657
    Acc: 0.529296875. Loss: 202.5321846822689. Val_Acc: 0.5957. Learning Rate: 0.0007352941
    Val Accuracy: 0.606
    ________________
    ________________
    Number 1281
    Mean A1: -0.98702432458707
    Mean Z1: -98.70243245870701
    Mean B1: -0.009016341315744657
    Acc: 0.5251953125. Loss: 205.77258231001738. Val_Acc: 0.5934999999999999. Learning Rate: 0.0007246377
    Val Accuracy: 0.588
    ________________
    ________________
    Number 1301
    Mean A1: -0.9947973971265537
    Mean Z1: -99.47973971265537
    Mean B1: -0.009016341315744657
    Acc: 0.5408203125. Loss: 209.0039658493883. Val_Acc: 0.5913999999999999. Learning Rate: 0.0007142857
    Val Accuracy: 0.592
    ________________
    ________________
    Number 1321
    Mean A1: -1.0025304818292513
    Mean Z1: -100.25304818292516
    Mean B1: -0.009016341315744657
    Acc: 0.528515625. Loss: 212.244850130526. Val_Acc: 0.591. Learning Rate: 0.0007042254
    Val Accuracy: 0.598
    ________________
    ________________
    Number 1341
    Mean A1: -1.0101864746344806
    Mean Z1: -101.01864746344803
    Mean B1: -0.009016341315744657
    Acc: 0.5291015625. Loss: 215.49415771398486. Val_Acc: 0.5896000000000001. Learning Rate: 0.0006944444
    Val Accuracy: 0.592
    ________________
    ________________
    Number 1361
    Mean A1: -1.0177482679085132
    Mean Z1: -101.77482679085134
    Mean B1: -0.009016341315744657
    Acc: 0.536328125. Loss: 218.72444725541862. Val_Acc: 0.5901000000000001. Learning Rate: 0.0006849315
    Val Accuracy: 0.616
    ________________
    ________________
    Number 1381
    Mean A1: -1.0252727212513926
    Mean Z1: -102.52727212513925
    Mean B1: -0.009016341315744657
    Acc: 0.55546875. Loss: 221.945425365234. Val_Acc: 0.5909. Learning Rate: 0.0006756757
    Val Accuracy: 0.626
    ________________
    ________________
    Number 1401
    Mean A1: -1.0327416915522587
    Mean Z1: -103.27416915522588
    Mean B1: -0.009016341315744657
    Acc: 0.56484375. Loss: 225.18772485264722. Val_Acc: 0.5938. Learning Rate: 0.0006666667
    Val Accuracy: 0.614
    ________________
    ________________
    Number 1421
    Mean A1: -1.0401473763796856
    Mean Z1: -104.01473763796854
    Mean B1: -0.009016341315744657
    Acc: 0.54140625. Loss: 228.42306699351244. Val_Acc: 0.5959. Learning Rate: 0.0006578947
    Val Accuracy: 0.564
    ________________
    ________________
    Number 1441
    Mean A1: -1.0475052623960999
    Mean Z1: -104.75052623960998
    Mean B1: -0.009016341315744657
    Acc: 0.51171875. Loss: 231.66291649255396. Val_Acc: 0.5934. Learning Rate: 0.0006493506
    Val Accuracy: 0.554
    ________________
    ________________
    Number 1461
    Mean A1: -1.054789951087271
    Mean Z1: -105.47899510872709
    Mean B1: -0.009016341315744657
    Acc: 0.5271484375. Loss: 234.8891698868017. Val_Acc: 0.5918000000000001. Learning Rate: 0.0006410256
    Val Accuracy: 0.586
    ________________
    ________________
    Number 1481
    Mean A1: -1.0620031510984722
    Mean Z1: -106.20031510984724
    Mean B1: -0.009016341315744657
    Acc: 0.542578125. Loss: 238.11509193488533. Val_Acc: 0.5908. Learning Rate: 0.0006329114
    Val Accuracy: 0.592
    ________________
    ________________
    Number 1501
    Mean A1: -1.0691770488451484
    Mean Z1: -106.91770488451485
    Mean B1: -0.009016341315744657
    Acc: 0.55078125. Loss: 241.3412648828543. Val_Acc: 0.5914. Learning Rate: 0.000625
    Val Accuracy: 0.614
    ________________
    ________________
    Number 1521
    Mean A1: -1.0763071291631368
    Mean Z1: -107.63071291631364
    Mean B1: -0.009016341315744657
    Acc: 0.5427734375. Loss: 244.57281432390792. Val_Acc: 0.5906. Learning Rate: 0.000617284
    Val Accuracy: 0.594
    ________________
    ________________
    Number 1541
    Mean A1: -1.083402597885484
    Mean Z1: -108.3402597885484
    Mean B1: -0.009016341315744657
    Acc: 0.5466796875. Loss: 247.79827141700534. Val_Acc: 0.5917000000000001. Learning Rate: 0.0006097561
    Val Accuracy: 0.606
    ________________
    ________________
    Number 1561
    Mean A1: -1.090437005480775
    Mean Z1: -109.04370054807751
    Mean B1: -0.009016341315744657
    Acc: 0.544140625. Loss: 251.0285277281535. Val_Acc: 0.594. Learning Rate: 0.0006024096
    Val Accuracy: 0.604
    ________________
    ________________
    Number 1581
    Mean A1: -1.0973573800353682
    Mean Z1: -109.73573800353681
    Mean B1: -0.009016341315744657
    Acc: 0.54765625. Loss: 254.23178819931113. Val_Acc: 0.5959999999999999. Learning Rate: 0.0005952381
    Val Accuracy: 0.614
    ________________
    ________________
    Number 1601
    Mean A1: -1.1042094210749256
    Mean Z1: -110.42094210749256
    Mean B1: -0.009016341315744657
    Acc: 0.551171875. Loss: 257.42655527444816. Val_Acc: 0.597. Learning Rate: 0.0005882353
    Val Accuracy: 0.602
    ________________
    ________________
    Number 1621
    Mean A1: -1.1110219665720598
    Mean Z1: -111.10219665720595
    Mean B1: -0.009016341315744657
    Acc: 0.546484375. Loss: 260.61927604542564. Val_Acc: 0.5985. Learning Rate: 0.0005813953
    Val Accuracy: 0.62
    ________________
    ________________
    Number 1641
    Mean A1: -1.1178032435107113
    Mean Z1: -111.78032435107114
    Mean B1: -0.009016341315744657
    Acc: 0.568359375. Loss: 263.8188897082861. Val_Acc: 0.601. Learning Rate: 0.0005747126
    Val Accuracy: 0.638
    ________________
    ________________
    Number 1661
    Mean A1: -1.1245311759106524
    Mean Z1: -112.45311759106524
    Mean B1: -0.009016341315744657
    Acc: 0.578515625. Loss: 267.0237549731285. Val_Acc: 0.6022999999999998. Learning Rate: 0.0005681818
    Val Accuracy: 0.632
    ________________
    ________________
    Number 1681
    Mean A1: -1.1312052240529258
    Mean Z1: -113.12052240529259
    Mean B1: -0.009016341315744657
    Acc: 0.578515625. Loss: 270.1997048346817. Val_Acc: 0.604. Learning Rate: 0.0005617978
    Val Accuracy: 0.622
    ________________
    ________________
    Number 1701
    Mean A1: -1.1378232066531204
    Mean Z1: -113.78232066531206
    Mean B1: -0.009016341315744657
    Acc: 0.5732421875. Loss: 273.38525514617623. Val_Acc: 0.6053999999999999. Learning Rate: 0.0005555556
    Val Accuracy: 0.62
    ________________
    ________________
    Number 1721
    Mean A1: -1.144375580036527
    Mean Z1: -114.4375580036527
    Mean B1: -0.009016341315744657
    Acc: 0.575. Loss: 276.56165312586717. Val_Acc: 0.6066999999999999. Learning Rate: 0.0005494505
    Val Accuracy: 0.624
    ________________
    ________________
    Number 1741
    Mean A1: -1.150893601877098
    Mean Z1: -115.0893601877098
    Mean B1: -0.009016341315744657
    Acc: 0.564453125. Loss: 279.7443243301828. Val_Acc: 0.6081. Learning Rate: 0.0005434783
    Val Accuracy: 0.62
    ________________
    ________________
    Number 1761
    Mean A1: -1.1573821545315355
    Mean Z1: -115.73821545315354
    Mean B1: -0.009016341315744657
    Acc: 0.58359375. Loss: 282.9094810602905. Val_Acc: 0.6077. Learning Rate: 0.0005376344
    Val Accuracy: 0.608
    ________________
    ________________
    Number 1781
    Mean A1: -1.1637956522747144
    Mean Z1: -116.37956522747143
    Mean B1: -0.009016341315744657
    Acc: 0.5626953125. Loss: 286.09547805923114. Val_Acc: 0.6082. Learning Rate: 0.0005319149
    Val Accuracy: 0.636
    ________________
    ________________
    Number 1801
    Mean A1: -1.1701404296793776
    Mean Z1: -117.01404296793775
    Mean B1: -0.009016341315744657
    Acc: 0.600390625. Loss: 289.23592404334585. Val_Acc: 0.6093. Learning Rate: 0.0005263158
    Val Accuracy: 0.636
    ________________
    ________________
    Number 1821
    Mean A1: -1.176453172662087
    Mean Z1: -117.64531726620874
    Mean B1: -0.009016341315744657
    Acc: 0.5892578125. Loss: 292.3859327047914. Val_Acc: 0.6126. Learning Rate: 0.0005208333
    Val Accuracy: 0.63
    ________________
    ________________
    Number 1841
    Mean A1: -1.182771336765596
    Mean Z1: -118.2771336765596
    Mean B1: -0.009016341315744657
    Acc: 0.5716796875. Loss: 295.55917824100345. Val_Acc: 0.615. Learning Rate: 0.0005154639
    Val Accuracy: 0.602
    ________________
    ________________
    Number 1861
    Mean A1: -1.1890349384575232
    Mean Z1: -118.90349384575232
    Mean B1: -0.009016341315744657
    Acc: 0.55390625. Loss: 298.7292636539437. Val_Acc: 0.6159. Learning Rate: 0.0005102041
    Val Accuracy: 0.604
    ________________
    ________________
    Number 1881
    Mean A1: -1.1952502950959476
    Mean Z1: -119.52502950959475
    Mean B1: -0.009016341315744657
    Acc: 0.579296875. Loss: 301.87772100072925. Val_Acc: 0.6166. Learning Rate: 0.0005050505
    Val Accuracy: 0.606
    ________________
    ________________
    Number 1901
    Mean A1: -1.201451630105832
    Mean Z1: -120.14516301058322
    Mean B1: -0.009016341315744657
    Acc: 0.5673828125. Loss: 305.043212437928. Val_Acc: 0.6167. Learning Rate: 0.0005
    Val Accuracy: 0.616
    ________________
    ________________
    Number 1921
    Mean A1: -1.2075837923992239
    Mean Z1: -120.75837923992238
    Mean B1: -0.009016341315744657
    Acc: 0.5751953125. Loss: 308.20419798733946. Val_Acc: 0.6180999999999999. Learning Rate: 0.0004950495
    Val Accuracy: 0.622
    ________________
    ________________
    Number 1941
    Mean A1: -1.2136704974658716
    Mean Z1: -121.36704974658717
    Mean B1: -0.009016341315744657
    Acc: 0.5791015625. Loss: 311.3505924399693. Val_Acc: 0.6197999999999999. Learning Rate: 0.0004901961
    Val Accuracy: 0.64
    ________________
    ________________
    Number 1961
    Mean A1: -1.2197212863079803
    Mean Z1: -121.97212863079805
    Mean B1: -0.009016341315744657
    Acc: 0.5826171875. Loss: 314.48385638428556. Val_Acc: 0.6216. Learning Rate: 0.0004854369
    Val Accuracy: 0.64
    ________________
    ________________
    Number 1981
    Mean A1: -1.225741377040138
    Mean Z1: -122.5741377040138
    Mean B1: -0.009016341315744657
    Acc: 0.587890625. Loss: 317.608374285768. Val_Acc: 0.6213000000000001. Learning Rate: 0.0004807692
    Val Accuracy: 0.608
    ________________
    Done
    ***********************************
    FINAL TEST ACCURACY OF: 0.5599
    


```python
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


```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



    
![png](output_6_3.png)
    



```python
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

```


    
![png](output_7_0.png)
    



```python

```




    7




```python

```




    7




```python

```

A1


```python

```


```python

```


```python


```


```python

```


```python

```
