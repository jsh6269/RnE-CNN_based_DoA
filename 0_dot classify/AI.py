import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )
# sd: 표준편차 standard deviation
# truncnorm => 표준화필요??

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T # ndmin : 최소크기를 지정...?
        target_vector = np.array(target_vector, ndmin=2).T  # .T는 전치행렬을 의미함.

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [self.bias]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network) # tmp is delta
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)              # learning_rate*delta*xi
        self.weights_hidden_out += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def visualize(self, fig):
        plt.figure(fig)
        for i in range(n):
            if (labels[i][0]):
                plt.scatter(data[i][0], data[i][1], c="red", marker='x', s=90)
            else:
                plt.scatter(data[i][0], data[i][1], c="blue", s=90)
        mylist = np.arange(-6.0, 8.0, 0.1)
        for i in mylist:
            initiation = simple_network.run((i, -6))
            check = 1 if initiation[0][0] - initiation[1][0] > 0 else 0
            for j in mylist:
                temp = simple_network.run((i, j))
                if (temp[0][0] - temp[1][0] > 0 and check==0):
                    plt.scatter(i, j, s=5, c="black")
                    break
                if (temp[0][0] - temp[1][0] < 0 and check==1):
                    plt.scatter(i, j, s=5, c="black")
                    break
                #if (abs(temp[0][0] - temp[1][0]) < 0.05):
                    #plt.scatter(i, j, c="black")
"""
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3),
          (-2, 6), (0, 4), (0, 4.5), (2, 0), (4, -2), (6, -2), (5, 1.8), (-2, 5.3)]
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8), (-4.3, 2.5), (-4, 0), (2, -6.1)
          , (-4,0.537), (-5.85, 3.45), (4, -6), (0, -2.34), (-6.3, -3.56)]
"""
"""
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3),
          (-2, 6), (0, 4), (0, 4.5), (5, 1.8), (-2, 5.3),
            (-4, 6), (8.01, -1.8), (6, 1)
          ] # red
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8), (2, -6.1)
          , (4, -6), (0, -2.34), (-6.3, -3.56), (-4.2, 2.1)
          , (-5, 0.2), (4, -3), (2, -3.5), (6, -2), (-2, -0.22)
          , (-6, 4), (-6.1, 3.5), (-2.3, 2), (-2, 0.88), (2 ,0)
          ]
"""
class1 = [(3, 4), (4.2, 5.3),  (6, 5),  (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3),
           (0, 4), (0, 4.5), (5, 1.8), (-2, 5.3),
            (-4, 6), (8.01, -1.8), (6, 1), (-4.2, 2.1), (-2, -0.22), (-2.3, 2)
    , (4, 3), (2, -3.5)

    , (-4, -5.6),(-6.3, -3.56)
          ] # red
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8), (2, -6.1)
          , (4, -6), (0, -2.34)
          , (-5, 0.2), (4, -3), (6, -2)
          , (-6, 4), (-6.1, 3.5), (2 ,0)
            ,(4, 6), (-2,6), (-2, 0.88)
          ]
labeled_data = []
n = len(class1)+len(class2)
for el in class1:
    labeled_data.append([el, [1, 0]])
for el in class2:
    labeled_data.append([el, [0, 1]])
print(labeled_data)
data, labels = zip(*labeled_data)
labels = np.array(labels)
data = np.array(data)
simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=2,
                               no_of_hidden_nodes=10,
                               learning_rate=0.1,
                               bias=None)
fig=1
for num in range(300):
    for i in range(len(data)):
        simple_network.train(data[i], labels[i])
    if(num%20==0):
        simple_network.visualize(fig)
        fig+=1
plt.show()
"""
for i in range(len(data)):
    print(labels[i])
    print(simple_network.run(data[i]))

for i in range(n):
    for j in range(n):
        print(-i,j)
        print(simple_network.run((-i,j)))
"""
"""
for i in range(n):
    for j in range(n):
        temp=simple_network.run((-i,j))
        if(abs( temp[0][0] - temp[1][0] ) < 0.1):
            print(-i,j)
            print(temp)
        temp=simple_network.run((i,-j))
        if(abs( temp[0][0] - temp[1][0] ) < 0.1):
            print(i,-j)
            print(temp)
"""

