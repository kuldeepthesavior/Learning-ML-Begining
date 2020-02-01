# 1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

# 2
###Start code here
img = mpimg.imread('bird.png')
data = img.reshape(1,194,259,3)
###End code

print(type(img))
print("Image dimension ",img.shape)
print(img.shape)
print("input data dimension ", data.shape)

# 3

plt.imshow(data[0,:,:,:])

# 4

graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(1)
    input_ = tf.constant(data.astype(np.float32))  ##The input data is coverted into tensor of type float32
    ###Start code here
    W = tf.Variable(tf.random_normal([5, 5, 3, 32]))
    b = tf.Variable(tf.random_normal([32]))

    conv = tf.nn.conv2d(input=input_, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    conv_bias = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_bias)
    conv_pool = tf.nn.pool(input=conv_out, window_shape=[3, 3], padding='VALID', pooling_type='MAX')
    ###ENd code


#  5

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    filters = sess.run(W)

    conv_output = sess.run(conv_out)

    after_pooling = sess.run(conv_pool)

###sanity check
print(conv_out)
print(conv_pool)
print(conv_output[0, 100:105, 200:205, 7])
print("\n", after_pooling[0, 100:105, 200:205, 7])

with open("output.txt", "w+") as file:
    file.write("mean1 = %f" % np.mean(conv_output))
    file.write("\nmean2 = %f" % np.mean(after_pooling))



# 6

def show_conv_results(data, title):
    fig1 = plt.figure()
    fig1.suptitle(title, fontsize=30)
    rows, cols = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        ax1 = fig1.add_subplot(rows, cols, i + 1)
        ax1.imshow(img, interpolation='none')
        ax1.axis('off')


def show_weights(W, title):
    fig2 = plt.figure()
    fig2.suptitle(title, fontsize=30)
    rows, cols = 4, 8
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        ax2 = fig2.add_subplot(rows, cols, i + 1)
        ax2.imshow(img, interpolation='none')
        ax2.axis('off')


show_weights(filters, title="filters, " + "shape:" + str(filters.shape))
show_conv_results(conv_output, title="after_convolution, " + "shape:" + str(conv_output.shape))
show_conv_results(after_pooling, title="after_pooling, " + "shape:" + str(after_pooling.shape))

