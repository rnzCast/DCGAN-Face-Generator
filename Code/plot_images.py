#%% ------------------------------------------- IMPORT PACKAGES --------------------------------------------------------
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

# code lines: 16
#%% ------------------------------------------- GENERATE FAKE FACES ----------------------------------------------------
# Renzo
def plot_fake_img(epoch, samples):
    fig, axes = plt.subplots(figsize=(16, 4),
                             nrows=2,
                             ncols=8,
                             sharey=True,
                             sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) * 255 / 2).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32, 32, 3)))
    plt.show()

#%% ------------------------------------------- LOAD SAMPLES -----------------------------------------------------------
with open('trained_model.pkl', 'rb') as f:
    samples = pkl.load(f)

plot_fake_img(-1, samples)



