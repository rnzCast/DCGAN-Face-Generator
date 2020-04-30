#%% ------------------------------------------- GENERATE FAKE FACES ----------------------------------------------------
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


#%% ------------------------------------------- PLOT REAL FACES --------------------------------------------------------
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

data_iter = iter(train_loader)
images, _ = data_iter.next()

fig = plt.figure(figsize=(20, 4))
plot_size = 20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])


#%% ------------------------------------------- DISCRIMINATOR ----------------------------------------------------------
def disc_conv(input_c, output, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    con = nn.Conv2d(input_c, output, kernel_size, stride, padding, bias=False)
    layers.append(con)
    if batch_norm:
        layers.append(nn.BatchNorm2d(output))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.layer_1 = disc_conv(3, conv_dim, 4, batch_norm=False)
        self.layer_2 = disc_conv(conv_dim, conv_dim * 2, 4)
        self.layer_3 = disc_conv(conv_dim * 2, conv_dim * 4, 4)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        x = F.leaky_relu(self.layer_3(x))
        x = x.view(-1, self.conv_dim * 4 * 4 * 4)
        x = self.fc(x)
        return x

#%% ------------------------------------------- PLOT LOSS ------------------------------------------------------------
# Renzo
fig, ax = plt.subplots()
loss = np.array(loss)
plt.plot(loss.T[0], label='Discriminator', alpha=0.5)
plt.plot(loss.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()


