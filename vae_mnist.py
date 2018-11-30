from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.contrib.layers import fully_connected as fc
import matplotlib.pyplot as plt
#TODO: Add requirements.txt
'''
versions of
datetime
numpy
tensorflow
tensorflow_probability
matplotlib
'''

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
tfd = tfp.distributions

'''Load and Normalise data'''
mnist = tf.keras.datasets.mnist
w = h = 28
image_px = w * h
(x_train, _),(x_test, _) = mnist.load_data()
[temp_max, temp_min] = [x_train.max(), x_train.min()]
x_train = np.reshape(x_train, (-1, image_px))
x_train = (x_train - temp_min) / temp_max #normalise to [0,1]
# print(x_train.max(), x_train.min())
[temp_max, temp_min] = [x_test.max(), x_test.min()]
x_test = np.reshape(x_test, (-1, image_px))
x_test = (x_test - temp_min)/temp_max #normalise to [0,1]
# print(x_test.max(), x_test.min())

input_ = tf.placeholder(tf.float32,[None, image_px], name='Input')
dim_z = 10 #dimensions of embedding vector
learning_rate=tf.constant(1e-5, tf.float32)
batch_size = 64
beta = 1

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

'''Encoder'''
with tf.variable_scope('encoding'):
    f1 = fc(input_, 512, scope='enc_fc1', activation_fn=tf.nn.relu)
    f2 = fc(f1, 384, scope='enc_fc2', activation_fn=tf.nn.relu)
    f3 = fc(f2, 256, scope='enc_fc3', activation_fn=tf.nn.relu)
    z_mu = fc(f3, dim_z, scope='Z_Mu', activation_fn=None)
    z_log_sigma_sq = fc(f3, dim_z, scope='Z_LogOf_SigmaSq', activation_fn=None) # use log(sig^2) instead of log(sigma)
    # Now dim_z mu's and sigma's, one for each dimension of z
    #latent space

    # encoded_dist = tf.distributions.Normal(loc=z_mu, scale=tf.sqrt(tf.exp(z_log_sigma_sq)))
    encoded_dist = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.sqrt(tf.exp(z_log_sigma_sq)))
    encoded = encoded_dist.sample() #sampled embedding

'''Decoder'''
with tf.variable_scope('decoding'):
    g1 = fc(encoded, 256, scope='dec_fc1', activation_fn=tf.nn.relu)
    g2 = fc(g1, 384, scope='dec_fc2', activation_fn=tf.nn.relu)
    g3 = fc(g2, 512, scope='dec_fc3', activation_fn=tf.nn.relu)
    # g3 = tf.Print(g3, ["Tensors input, f3, mu, log(var), z, g3", input_[0], f3[0], z_mu[0], z_log_sigma_sq[0], encoded[0], g3[0]])
    # g3 = tf.Print(g3, ["IsNaN? f3, mu, log(var), g3", tf.reduce_any(tf.is_nan(f3[0])), tf.reduce_any(tf.is_nan(z_mu[0])), tf.reduce_any(tf.is_nan(z_log_sigma_sq[0])), tf.reduce_any(tf.is_nan(g3[0]))])

    output_nonNormalised = fc(g3, image_px, scope='Non-normalisedOutput', activation_fn=None)
    # output_nonNormalised = tf.Print(output_nonNormalised, ["output_1_presigmoid", output_nonNormalised], summarize=10)
    output_ = tf.nn.sigmoid(output_nonNormalised, name='VAEOutput')
    # output_ = tf.Print(output_, ["output_", output_], summarize=3)
    # print(input_.shape, output_nonNormalised.shape)

''' Loss '''
with tf.variable_scope('loss'):
    # Reconstruction loss: Minimize the cross-entropy loss H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
    # recon_loss = -tf.nn.sigmoid_cross_entropy_with_logits(labels=input_, logits=output_nonNormalised, name='CrossEntropyLoss_x_xhat')
    # recon_loss = tf.reduce_mean(recon_loss)
    # # Reconstruction loss: Minimize the L2-norm loss
    # recon_loss = tf.losses.mean_squared_error(input_, output_,scope='ReconstructionLoss')

    #Reconstruction loss: Minimize cross entropy loss
    # recon_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(input_),logits=output_nonNormalised)
    # recon_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_,logits=output_nonNormalised)
    epsilon = 1e-10
    recon_loss = tf.reduce_mean(-tf.reduce_sum(input_ * tf.log(epsilon+output_) + (1-input_) * tf.log(epsilon+1-output_), axis=1))

    # Latent loss
    # Kullback Leibler divergence: measure the difference between z_posterior_approximated and z_prior
    z_prior = tfd.MultivariateNormalDiag(loc=np.zeros(dim_z, dtype=np.float32), scale_diag=np.ones(dim_z, dtype=np.float32)) # normal distribution in R^{dim_z} space
    # kl_divergence(approx_posterior, latent_prior)
    latent_loss = tf.reduce_mean(tf.distributions.kl_divergence(encoded_dist, z_prior))
    # latent_loss = -0.5 * tf.reduce_sum(1 + tf.square(tf.log(z_log_sigma_sq)) - tf.square(z_mu) - z_log_sigma_sq, axis=1)
    # latent_loss = tf.reduce_mean(latent_loss) # go from (?, 10) to scalar per batch

    # Total loss = sum of both losses
    loss = recon_loss  + beta * latent_loss

cost = tf.reduce_mean(loss)
global_step = tf.train.get_or_create_global_step()
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# MeanG3_Summary = tf.summary.scalar('Mean_G3_Layer', tf.reduce_mean(g3))
Recon_loss_Summary = tf.summary.scalar(name='Reconstruction_Loss', tensor=recon_loss)
Latent_loss_Summary = tf.summary.scalar(name='Latent_Loss', tensor=latent_loss)
Total_loss_Summary = tf.summary.scalar(name='Total_Loss', tensor=loss)
original_image = tf.summary.image(name='Input_Image', tensor=tf.manip.reshape(input_, [batch_size, 28,28, 1]))
reconstructed_image = tf.summary.image(name='Reconstructed_Image', tensor=tf.manip.reshape(output_, [batch_size, 28,28, 1]))

'''Training'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss = []
valid_loss = []

epoch_printing = 5
epochs = 100

now = datetime.now()
logdir = "./Logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
all_summaries = tf.summary.merge_all()

for e in range(epochs):
    print("Now starting epoch",e)
    total_batch = int(len(x_train) / batch_size)
    for ibatch in range(total_batch):

        # choose next batch for train and random batch for test
        # print("Example range", ibatch*batch_size, (ibatch+1)*batch_size)
        train_batch = x_train[ibatch*batch_size:(ibatch+1)*batch_size]
        indices = np.random.choice(range(len(x_test)), batch_size, replace=False) # randomly choose from test set
        test_batch = x_test[indices, :]

        my_global_step = sess.run([global_step], feed_dict={input_: train_batch})
        # print('Global step', my_global_step)

        if my_global_step[0] % 10 == 0:
            batch_cost, derp, summary = sess.run([cost, opt, all_summaries], feed_dict={input_: train_batch})
            step = tf.train.global_step(sess, tf.train.get_global_step())
            writer.add_summary(summary=summary, global_step=step)
            writer.flush()
        else:
            batch_cost, _ = sess.run([cost, opt], feed_dict={input_: train_batch})

        batch_cost_test = sess.run(cost, feed_dict={input_: test_batch})

    if (e + 1) % epoch_printing == 0:
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost),
              "Validation loss: {:.4f}".format(batch_cost_test))

    loss.append(batch_cost)
    valid_loss.append(batch_cost_test)

plt.plot(range(epochs), loss, 'bo', label="Training_loss")
plt.plot(range(epochs), valid_loss, 'r', label='Validation_loss')
plt.legend('Training_loss', 'Validation_loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.show()

# Test the trained model: generation
# Sample noise vectors from N(0, 1)
z = np.random.normal(size=[batch_size, dim_z])
x_generated = sess.run(output_, feed_dict={encoded: z})

n = np.sqrt(batch_size).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(28, 28)

fig = plt.figure()
plt.imshow(I_generated, cmap='gray')
plt.savefig('I_generated.png')
plt.close(fig)