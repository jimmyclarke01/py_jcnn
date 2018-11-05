import jcnn
import numpy as np

xor_data = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

xor_labels = np.array([0,
                       1,
                       1,
                       0])

count_data = xor_data

count_labels = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1]])



def perform_test(nn):
    n = np.random.randint(0, 4)
    input = count_data[n]
    label = count_labels[n]

    output = nn.forwardProp(input)
    nn.backProp(output, label)
    if np.argmax(output) == n :
        score = 1
    else:
        score = 0

    print(score)
    return score


nn = jcnn.NN(2)

layer1 = jcnn.NNLayer(2, 2, jcnn.Activations().logistic)
layer2 = jcnn.NNLayer(2, 3, jcnn.Activations().logistic)

nn.addLayer(layer1)
nn.addLayer(layer2)

score = 0
for i in np.arange(1000):
    score += perform_test(nn)

print(score/10)