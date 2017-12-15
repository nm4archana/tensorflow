import tensorflow as tf
import numpy as np
import pandas as pd


# First layer
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
Define a placeholder
'''
data = tf.placeholder(tf.float32, [None, 784])

'''
No.of layers is initialized as 500
'''
layer_1_nodes = 500



'''
Distinct classes of y
'''
y_distinct_classes = 10;

'''
No. of attributes of input data
'''
x_dimension = 784;

'''
First layer : 100 Nodes
'''
weight1 = tf.Variable(tf.random_normal([784, layer_1_nodes]))
#bias1 = tf.Variable(tf.random_normal([layer_1_nodes]))
bias1 = tf.Variable(tf.zeros([layer_1_nodes]))
layer1 = tf.matmul(data,weight1) + bias1;
layer1 = tf.nn.relu(layer1)

'''
Output
'''
weight2 = tf.Variable(tf.random_normal([layer_1_nodes,y_distinct_classes]))
#bias2 = tf.Variable(tf.random_normal([layer_3_nodes]))
bias2 = tf.Variable(tf.zeros([y_distinct_classes]))
layer2 = tf.matmul(layer1,weight2) + bias2;

y = layer2

'''
Creating a placeholder for y
'''
y_ = tf.placeholder(tf.float32,[None,y_distinct_classes])

'''
Calculating the cost function
'''
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))

'''
Minimizing the cost function - By using an optimizer
'''
#optimizer = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

'''
Initiate an interactive session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

'''
Number of epochs is  set to 10000
'''
n_epoch = 10000

for _ in range(n_epoch):
    trainX , trainy = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={data:trainX,y_: trainy})
    '''
        Printing accuracy, cross entropy for train and test for every 1000 epochs 
    '''
    if _ % 1000 == 0:

        '''
        Finding the predicted correct labels 
        '''
        predicted_valid_labels = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        '''
        Finding the accuracy of the predicted labels
        '''
        accuracy = tf.reduce_mean(tf.cast(predicted_valid_labels,tf.float32))

        '''
        Printing the values
        '''
        #acc = sess.run(accuracy, feed_dict={data: mnist.test.images, y_: mnist.test.labels})
        acc = sess.run(accuracy, feed_dict={data: mnist.test.images, y_: mnist.test.labels})
        print("Accuracy: ",acc)

        cost_train = sess.run(cross_entropy,feed_dict={data: trainX, y_: trainy})
        print("Cross Entropy Train: ", cost_train)

        #cost_test = sess.run(cross_entropy, feed_dict={data: mnist.test.images, y_: mnist.test.labels})
        cost_test = sess.run(cross_entropy, feed_dict={data: mnist.test.images, y_:mnist.test.labels})
        print("Cross Entropy Test: ", cost_test)

'''
Prediction and accuracy for complete epochs
'''
predicted_valid_labels = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predicted_valid_labels,tf.float32))
#cost_test = sess.run(cross_entropy, feed_dict={data: mnist.test.images, y_: mnist.test.labels})
cost_test = sess.run(cross_entropy, feed_dict={data: mnist.test.images, y_: mnist.test.labels})
print("Cross Entropy: ", cost_test)
