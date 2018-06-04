import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import mnist
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
from gan import build_generator, build_discriminator, plot_images, make_trainable, get_session


log_dir="."
KTF.set_session(get_session())  # Allows 2 jobs per GPU, Please do not change this during the tutorial

# prepare MNIST dataset
data = mnist.load_data()
X_train = data.train_images.reshape(-1, 28, 28, 1) / 255.
X_test = data.test_images.reshape(-1, 28, 28, 1) / 255.

# plot some real images
idx = np.random.choice(len(X_train), 16)
plot_images(X_train[idx], fname=log_dir + '/real_images.png')

# --------------------------------------------------
# Set up generator, discriminator and GAN (stacked generator + discriminator)
# Feel free to modify eg. :
# - the provided models (see gan.py)
# - the learning rate
# - the batchsize
# --------------------------------------------------

# Set up generator
print('\nGenerator')
latent_dim = 100
generator = build_generator(latent_dim)
print(generator.summary())

# Set up discriminator
print('\nDiscriminator')
discriminator = build_discriminator()
print(discriminator.summary())
d_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

# Set up GAN by stacking the discriminator on top of the generator
print('\nGenerative Adversarial Network')
gan_input = Input(shape=[latent_dim])
gan_output = discriminator(generator(gan_input))
GAN = Model(gan_input, gan_output)
print(GAN.summary())
g_opt = Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
GAN.compile(loss='binary_crossentropy', optimizer=g_opt)
# Compile saves the trainable status of the model --> After the model is compiled, updating using make_trainable will have no effect

# --------------------------------------------------
# Pretrain the discriminator:
# --------------------------------------------------

# - Create a dataset of 10000 real train images and 10000 fake images.
ntrain = 10000
no = np.random.choice(60000, size=ntrain, replace='False')
real_train = X_train[no,:,:,:]   # sample real images from training set
noise_gen = np.random.uniform(0,1,size=[ntrain, latent_dim])
generated_images = generator.predict(noise_gen)  # generate fake images with untrained generator
print(generated_images.shape)
X = np.concatenate((real_train, generated_images))
y = np.zeros([2*ntrain, 2])   # class vector: one-hot encoding
y[:ntrain, 1] = 1             # class 1 for real images
y[ntrain:, 0] = 1             # class 0 for generated images

# - Train the discriminator for 1 epoch on this dataset.
discriminator.fit(X,y, epochs=1, batch_size=128)

# - Create a dataset of 5000 real test images and 5000 fake images.
no = np.random.choice(10000, size=ntrain//2, replace='False')
real_test = X_test[no,:,:,:]   # sample real images from test set
noise_gen = np.random.uniform(0,1,size=[ntrain//2, latent_dim])
generated_images = generator.predict(noise_gen)    # generate fake images with untrained generator
Xt = np.concatenate((real_test, generated_images))
yt = np.zeros([ntrain, 2])   # class vector: one-hot encoding    
yt[:ntrain//2, 1] = 1         # class 1 for real images
yt[ntrain//2:, 0] = 1         # class 0 for generated images

# - Evaluate the test accuracy of your network.
pretrain_loss, pretrain_acc = discriminator.evaluate(Xt, yt, verbose=0, batch_size=128)
print('Test accuracy: %04f' % pretrain_acc)

# loss vector
losses = {"d":[], "g":[]}
discriminator_acc = []

# main training loop
def train_for_n(epochs=1, batch_size=32):
    
    for epoch in range(epochs):
        
        # Plot some fake images   
        noise = np.random.uniform(0.,1.,size=[16,latent_dim])
        generated_images = generator.predict(noise)
        plot_images(generated_images, fname=log_dir + '/generated_images_' + str(epoch))

        iterations_per_epoch = 60000//batch_size    # number of training steps per epoch
        perm = np.random.choice(60000, size=60000, replace='False')
        
        for i in range(iterations_per_epoch):
            
            # Create a mini-batch of data (X: real images + fake images, y: corresponding class vectors)
            image_batch = X_train[perm[i*batch_size:(i+1)*batch_size],:,:,:]    # real images   
            noise_gen = np.random.uniform(0.,1.,size=[batch_size, latent_dim])
            generated_images = generator.predict(noise_gen)                     # generated images
            X = np.concatenate((image_batch, generated_images))
            y = np.zeros([2*batch_size,2])   # class vector
            y[0:batch_size,1] = 1
            y[batch_size:,0] = 1
            
            # Train the discriminator on the mini-batch
            d_loss, d_acc  = discriminator.train_on_batch(X,y)
            losses["d"].append(d_loss)
            discriminator_acc.append(d_acc)

            # Create a mini-batch of data (X: noise, y: class vectors pretending that these produce real images)
            noise_tr = np.random.uniform(0.,1.,size=[batch_size,latent_dim])
            y2 = np.zeros([batch_size,2])
            y2[:,1] = 1

            # Train the generator part of the GAN on the mini-batch
            g_loss = GAN.train_on_batch(noise_tr, y2)
            losses["g"].append(g_loss)

train_for_n(epochs=10, batch_size=128)

# - Plot the loss of discriminator and generator as function of iterations
plt.figure(figsize=(10,8))
plt.semilogy(losses["d"], label='discriminitive loss')
plt.semilogy(losses["g"], label='generative loss')
plt.legend()
plt.savefig(log_dir + '/loss.png')

# - Plot the accuracy of the discriminator as function of iterations
plt.figure(figsize=(10,8))
plt.semilogy(discriminator_acc, label='discriminator accuracy')
plt.legend()
plt.savefig(log_dir + '/discriminator_acc.png')

