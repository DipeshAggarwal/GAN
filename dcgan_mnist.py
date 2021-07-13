from core.nn.conv import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50, help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size for training")
args = vars(ap.parse_args())

NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]

print("[INFO] Loading MNIST Dataset...")
(train_x, _), (test_x, _) = mnist.load_data()
train_images = np.concatenate([train_x, test_x])

train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype(float) - 127.5) / 127.5

print("[INFO] Building Generator")
gen = DCGAN.build_generator(7, 64, channels=1)

print("[INFO] Building Discriminator")
disc = DCGAN.build_discriminator(28, 28, 1)
disc_opt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002/NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=disc_opt)

print("[INFO] Building GAN...")
disc.trainable = False
gan_input = Input(shape=(100,))
gan_output = disc(gen(gan_input))
gan = Model(gan_input, gan_output)

gan_opt = Adam(lr=0.0002, beta_1=0.5, decay=0.0002/NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=gan_opt)

print("[INFO] Started Training...")
benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

for epoch in range(NUM_EPOCHS):
	print("[INFO] Starting Epoch {} of {}".format(epoch+1, NUM_EPOCHS))
	batches_per_epochs = int(train_images.shape[0] / BATCH_SIZE)

	for i in range(0, batches_per_epochs):
		p = None

		image_batch = train_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

		gen_images = gen.predict(noise, verbose=0)

		X = np.concatenate((image_batch, gen_images))
		y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
		X, y = shuffle(X, y)

		disc_loss = disc.train_on_batch(X, y)

		noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
		gan_loss = gan.train_on_batch(noise, [1] * BATCH_SIZE)

		if i == batches_per_epochs - 1:
			p = [args["output"], "epoch_{}_output.png".format(str(epoch+1).zfill(4))]

		else:
			if epoch < 10 and i % 25 == 0:
				p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch+1).zfill(4), str(i).zfill(5))]

			elif epoch >= 10 and i % 100 == 0:
				p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch+1).zfill(4), str(i).zfill(5))]


		if p is not None:
			print("[INFO] Step {}_{}: Discriminator Loss = {:.6f}, Adversarial Loss = {:.6f}".format(epoch + 1, i, disc_loss, gan_loss))

			images = gen.predict(benchmark_noise)
			images = ((images * 127.5) + 127.5).astype("uint8")
			images = np.repeat(images, 3, axis=-1)
			vis = build_montages(images, (28, 28), (16, 16))[0]

			p = os.path.sep.join(p)
			cv2.imwrite(p, vis)
