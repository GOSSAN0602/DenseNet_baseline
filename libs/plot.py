import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss, n_epochs, SAVE_PATH):
    fig = plt.figure()
    x = np.arange(n_epochs)
    plt.plot(x, loss, label='train_loss')
    plt.legend()
    plt.savefig(SAVE_PATH+'loss.png') 
