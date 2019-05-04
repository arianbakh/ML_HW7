import math
import numpy as np
import os
import sys
from matplotlib import pyplot as plt


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
V_PATH = os.path.join(BASE_DIR, 'v')
W_PATH = os.path.join(BASE_DIR, 'w')


# Algorithm Parameters
ALPHA = 0.02


_binary_sigmoid = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
_binary_sigmoid_derivative = np.vectorize(lambda x: _binary_sigmoid(x) * (1 - _binary_sigmoid(x)))


def _get_pairs():
    a = [0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 0, 1, 0, 1, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    c = [0, 0, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    d = [1, 1, 1, 1, 0, 0, 0,
         0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 1, 0, 0,
         1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    e = [1, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0]
    f = [1, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    g = [0, 0, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 1, 1, 1,
         1, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 0, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
    h = [1, 1, 1, 0, 1, 1, 1,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         0, 1, 0, 0, 0, 1, 0,
         1, 1, 1, 0, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0]
    i = [0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0, 0]
    j = [0, 0, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 0, 1, 0, 0,
         0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]

    target_vectors = [a, b, c, d, e, f, g, h, i, j]
    input_vectors = [[1] + x for x in target_vectors]  # add bias

    return np.array(input_vectors), np.array(target_vectors)


def _init_v(input_vectors, hidden_layer_size):
    if os.path.exists(V_PATH):
        return np.load(V_PATH)
    else:
        v = np.random.rand(input_vectors.shape[1], hidden_layer_size) - 0.5
        np.save(V_PATH, v)
        return v


def _init_w(target_vectors, hidden_layer_size):
    if os.path.exists(W_PATH):
        return np.load(W_PATH)
    else:
        w = np.random.rand(hidden_layer_size, target_vectors.shape[1]) - 0.5
        np.save(W_PATH, w)
        return w


def _train(input_vectors, target_vectors, hidden_layer_size, threshold):
    v = _init_v(input_vectors, hidden_layer_size)
    w = _init_w(target_vectors, hidden_layer_size)

    epoch = 0
    all_correct = False
    while not all_correct:
        epoch += 1
        tse = 0

        for i in range(input_vectors.shape[0]):
            # feed forward
            x = np.reshape(input_vectors[i], (1, input_vectors[i].shape[0]))
            z_in = np.matmul(
                x,
                v
            )
            z = _binary_sigmoid(z_in)
            z[0, 0] = 1  # bias
            y_in = np.matmul(
                z,
                w
            )
            y = _binary_sigmoid(y_in)

            # error calculation
            tse += ((y - target_vectors[i]) ** 2).sum()
            threshold_vector = np.abs(y - target_vectors[i]) < threshold
            if np.all(threshold_vector):
                all_correct = True
                break  # to prevent further changes to weights

            # back propagation
            output_sigma = (target_vectors[i] - y) * _binary_sigmoid_derivative(y_in)
            delta_w = ALPHA * np.matmul(
                z.T,
                output_sigma
            )
            hidden_sigma_in = np.matmul(
                output_sigma,
                w.T
            )
            hidden_sigma = hidden_sigma_in * _binary_sigmoid_derivative(z_in)
            delta_v = ALPHA * np.matmul(
                x.T,
                hidden_sigma
            )

            # update weights
            w += delta_w
            v += delta_v

        if epoch % 10 == 0:
            sys.stdout.write('\rEpoch %d, Error %f' % (epoch, tse))
            sys.stdout.flush()

    print()  # newline

    return epoch


def run(threshold):
    input_vectors, target_vectors = _get_pairs()
    x = list(range(30, 9, -1))
    y = []
    for hidden_layer_size in x:
        total_epochs = _train(input_vectors, target_vectors, hidden_layer_size, threshold)
        y.append(total_epochs)

    plt.plot(x, y)
    plt.savefig(os.path.join(BASE_DIR, 'threshold_%f.png' % threshold))


if __name__ == '__main__':
    run(float(sys.argv[1]))
