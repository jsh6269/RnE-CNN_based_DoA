import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    )

class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes1,
                 no_of_hidden_nodes2,
                 learning_rate,
                 bias,
                 bias_node
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes1 = no_of_hidden_nodes1
        self.no_of_hidden_nodes2 = no_of_hidden_nodes2
        self.learning_rate = learning_rate
        self.bias = bias
        self.bias_node = bias_node
        self.create_weight_matrices()

    def create_weight_matrices(self):

        rad = 1 / np.sqrt(self.no_of_in_nodes + self.bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights1 = X.rvs((self.no_of_hidden_nodes1,
                                        self.no_of_in_nodes + self.bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes1 + self.bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights2 = X.rvs((self.no_of_hidden_nodes2,
                                         self.no_of_hidden_nodes1 + self.bias_node))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes2 + self.bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights3 = X.rvs((self.no_of_out_nodes,
                               self.no_of_hidden_nodes2 + self.bias_node))

    def train(self, input_vector, target_vector):

        if self.bias_node:
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights1, input_vector)
        sigmoid_output_vector1 = activation_function(output_vector1)

        if self.bias_node:
            sigmoid_output_vector1 = np.concatenate((sigmoid_output_vector1, [[self.bias]]))

        output_vector2 = np.dot(self.weights2, sigmoid_output_vector1)
        sigmoid_output_vector2 = activation_function(output_vector2)

        if self.bias_node:
            sigmoid_output_vector2 = np.concatenate((sigmoid_output_vector2, [[self.bias]]))

        output_vector3 = np.dot(self.weights3, sigmoid_output_vector2)
        sigmoid_output_vector3 = activation_function(output_vector3)

        # weight3 를 수정
        output_errors = target_vector - sigmoid_output_vector3
        delta1 = output_errors * sigmoid_output_vector3 * (1.0 - sigmoid_output_vector3)
        tmp = self.learning_rate * np.dot(delta1, sigmoid_output_vector2.T)
        self.weights3 += tmp

        # weight2 를 수정
        hidden_errors1 = np.dot(self.weights3.T, delta1)
        delta2 = hidden_errors1 * sigmoid_output_vector2 * (1.0 - sigmoid_output_vector2)
        delta2 = delta2[:-1] if self.bias_node else delta2
        tmp = self.learning_rate * np.dot(delta2, sigmoid_output_vector1.T)
        self.weights2 += tmp

        # weight1을 수정
        hidden_errors2 = np.dot(self.weights2.T, delta2)
        tmp = hidden_errors2 * sigmoid_output_vector1 * (1.0 - sigmoid_output_vector1)
        tmp = tmp[:-1] if self.bias_node else tmp
        tmp = np.dot(tmp, input_vector.T)
        self.weights1 += self.learning_rate * tmp

    def run(self, input_vector):

        if (self.bias_node):
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = np.dot(self.weights1, input_vector)
        output_vector = activation_function(output_vector)

        if (self.bias_node):
            output_vector = np.concatenate((output_vector, [[self.bias]]))

        output_vector = np.dot(self.weights2, output_vector)
        output_vector = activation_function(output_vector)

        if (self.bias_node):
            output_vector = np.concatenate((output_vector, [[self.bias]]))

        output_vector = np.dot(self.weights3, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


    def visualize(self, fig):
        plt.figure(fig)
        for i in range(n):
            if (labels[i][0]):
                plt.scatter(data[i][0], data[i][1], c="red", marker='x', s=90)
            else:
                plt.scatter(data[i][0], data[i][1], c="blue", s=90)
        mylist = np.arange(-6.0, 8.0, 0.05)
        for i in mylist:
            initiation = simple_network.run((i, -6))
            check = 1 if initiation[0][0] - initiation[1][0] > 0 else 0
            for j in mylist:
                temp = simple_network.run((i, j))
                determination=temp[0][0] - temp[1][0]
                if ((determination > 0 and check == 0) or (determination < 0 and check == 1)):
                    plt.scatter(i, j, s=5, c="black")
                    break  # 그래프를 그리는 시간을 단축하기 위해 코드를 이렇게 짰지만 break 없이 아래와 같이 짜는 것이 맞음
        """for i in mylist:
            initiation = simple_network.run((i, -6))
            check = 1 if initiation[0][0] - initiation[1][0] > 0 else 0
            for j in mylist:
                temp = simple_network.run((i, j))
                determination=temp[0][0] - temp[1][0]
                if ((determination > 0 and check == 0) or (determination < 0 and check == 1)):
                    plt.scatter(i, j, s=5, c="black")
                    check = 1 if check=0 else 0
        """
        for i in mylist:
            initiation = simple_network.run((-6, i))
            check = 1 if initiation[0][0] - initiation[1][0] > 0 else 0
            for j in mylist:
                temp = simple_network.run((j, i))
                determination=temp[0][0] - temp[1][0]
                if ((determination > 0 and check == 0) or (determination < 0 and check == 1)):
                    plt.scatter(j, i, s=5, c="black")
                    break
        #name = "C:/Users/sj/Desktop/graph/ AI3_with_bias %d" %(fig)
        name = "./data3/AI3_with_bias %lf rate %lf repeats %d.png" %(self.bias, self.learning_rate, fig)
        # plt.savefig(name, dpi=300)
        plt.show()
"""
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3)
          ] # red
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6)
          , (-6, 4), (-6.1, 3.5), (2, 0)
          ]
"""
class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
          (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3),
          (-2, 6), (0, 4), (0, 4.5), (5, 1.8), (-2, 5.3), (-2, 0.88),
            (-4, 6), (8.01, -1.8), (6, 1), (-4.2, 2.1), (-2, -0.22), (-2.3, 2)
          ] # red
class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
          (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
          (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8), (2, -6.1)
          , (4, -6), (0, -2.34), (-6.3, -3.56)
          , (-5, 0.2), (4, -3), (2, -3.5), (6, -2)
          , (-6, 4), (-6.1, 3.5), (2, 0)
          ]
"""
class1 = [(-3, 4), (-4.2, 5.3), (-4, 3), (-6, 5), (-4, 6), (-3.7, 5.8),
          (-6, -2)
          ] # red
class2 = [(3, -4), (2, -3.5), (1, -6), (3, -4.3), (4, -5.6), (-2, 0), (2, 4)
          ]
"""

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
                               no_of_hidden_nodes1=10,
                               no_of_hidden_nodes2=10,
                               learning_rate=0.01,
                               bias=1.0,
                               bias_node=1)

for num in range(5001):
    for i in range(len(data)):
        simple_network.train(data[i], labels[i])
    if(num%1000==0):
        simple_network.visualize(num)
plt.show()
plt.close()
