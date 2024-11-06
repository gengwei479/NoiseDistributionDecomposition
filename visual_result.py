import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def line_graphs_00(inputsX, inputsY, label, dir):
    pic_obj=plt.figure()#figsize=(4, 3), dpi=300
    fig = plt.subplot(111, facecolor = '#EBEBEB')
    for y_key, y_value in inputsY.items():
        fig.plot(inputsX, y_value, label = y_key, lw = 1, ls = '-', alpha = 1)    
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    plt.grid(c='w')
    plt.legend()
    # plt.show()
    pic_obj.savefig(dir)

def line_graphs_01(inputsX, inputsY, label, dir):
    pic_obj=plt.figure()#figsize=(4, 3), dpi=300
    fig = plt.subplot(111, facecolor = '#EBEBEB')
    for key, value in inputsY.items():
        x_data = []
        y_data = []
        for j in value:
            # assert len(inputsX) == len(j)
            inputsX = inputsX[:min(len(inputsX), len(j))]
            j = j[:min(len(inputsX), len(j))]
            x_data += inputsX
            if hasattr(j, 'tolist'):
                y_data += j.tolist()
            else:
                y_data += j
        data = pd.DataFrame({'x': x_data, 'y': y_data})
        sns.lineplot(data = data, x = 'x', y = 'y', label = key)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    plt.grid(c='w')
    plt.legend()
    # plt.show()
    pic_obj.savefig(dir)

# inputsX = np.arange(0, 60000, 1000)
# inputsY = {}
# inputsY["aaaa"] = np.random.rand(len(inputsX))
# inputsY["bbbb"] = np.random.rand(len(inputsX))
# label = ["iteration", "average reward"]
# line_graphs_00(inputsX, inputsY, label, 0)