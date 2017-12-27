import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hidden_layer1_size = 400
hidden_layer2_size = 400
hidden_layer3_size = 400

classes = 10
batch_size = 100

x = tf.placeholder('float', [None,784])
y = tf.placeholder('float')

def nn_model (data):
    hidden_layer1 = {'W':tf.Variable(tf.random_normal([784,hidden_layer1_size])), 'B':tf.Variable(tf.random_normal([hidden_layer1_size]))}
    hidden_layer2 = {'W':tf.Variable(tf.random_normal([hidden_layer1_size,hidden_layer2_size])),
                      'B':tf.Variable(tf.random_normal([hidden_layer2_size]))}
    hidden_layer3 = {'W':tf.Variable(tf.random_normal([hidden_layer2_size,hidden_layer3_size])),
                      'B':tf.Variable(tf.random_normal([hidden_layer3_size]))}
    output_layer = {'W':tf.Variable(tf.random_normal([hidden_layer3_size,classes])),
                      'B':tf.Variable(tf.random_normal([classes]))}

    layer1 = tf.add( tf.matmul(data,hidden_layer1['W']) , hidden_layer1['B'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add( tf.matmul(layer1,hidden_layer2['W']) , hidden_layer2['B'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add( tf.matmul(layer2,hidden_layer3['W']) , hidden_layer3['B'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add( tf.matmul(layer3,output_layer['W']) , output_layer['B'])

    return output




def train_nn(x):
    prediction = nn_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_nn(x)
