import numpy as np
from matplotlib import pyplot as plt

data = [[0.5, 1, 1.5, 2, 2.5, 3, 0], [2, 4, 6, 8, 10, 12, 1]]

mdata = [0.625, 1.25, 1.875, 2.5, 3.215, 3.75, 0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))
#sigmoid_p is de afgeleide van de sigmoid functie


def train():
    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    w5 = np.random.rand()
    w6 = np.random.rand()
    b = np.random.rand()

    aantalRuns = 600000
    learningRate = 0.04
    costs = []

    for i in range(aantalRuns):
        randomKeuze = np.random.randint(len(data))
        keuze = data[randomKeuze]

        z = keuze[0] * w1 + keuze[1] * w2 + keuze[2] * w3 + keuze[3] * w4 + keuze[4] * w5 + keuze[5] * w6 + b

        ver = sigmoid(z)
        val = keuze[6]

        cost = np.square(ver - val)


        if i % 100 == 0:
            c = 0
            for j in range(len(data)):
                p = data[j]
                p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
                c += np.square(p_pred - p[2])
            costs.append(c)
            print(cost)



        afCost_Pred = 2 * (ver - val)
        afPred_Z = sigmoid_p(z)

        af_w1 = keuze[0]
        af_w2 = keuze[1]
        af_w3 = keuze[2]
        af_w4 = keuze[3]
        af_w5 = keuze[4]
        af_w6 = keuze[5]
        af_b = 1

        afCost_z = afCost_Pred * afPred_Z
        #z op x-as, cost op y-as, hierboven is daar gewoon de afgeleide van, waarin afPred_Z de helling van z aangeeft en bij een piek is die natuurlijk 0, en als je iets keer 0 doet wordt er dus geen learning meer gedaan

        afCost_w1 = afCost_z * af_w1
        afCost_w2 = afCost_z * af_w2
        afCost_w3 = afCost_z * af_w3
        afCost_w4 = afCost_z * af_w4
        afCost_w5 = afCost_z * af_w5
        afCost_w6 = afCost_z * af_w6
        afCost_b = afCost_z * af_b

        w1 = w1 - learningRate * afCost_w1
        w2 = w2 - learningRate * afCost_w2
        w3 = w3 - learningRate * afCost_w3
        w4 = w4 - learningRate * afCost_w4
        w5 = w5 - learningRate * afCost_w5
        w6 = w6 - learningRate * afCost_w6
        b = b - learningRate * afCost_b



    

    return costs, w1, w2, w3, w4, w5, w6, b

costs, w1, w2, w3, w4, w5, w6, b = train()

fig = plt.plot(costs)


z = w1 * mdata[0] + w2 * mdata[1] + w3 * mdata[2] + w4 * mdata[3] + w5 * mdata[4] + w6 * mdata[5] + b
print(sigmoid(z))