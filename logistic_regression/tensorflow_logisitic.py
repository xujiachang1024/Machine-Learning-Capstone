import pandas as pd
import numpy as np
import tensorflow as tf



def sigmoid(z):
    return 1/(1+np.exp(-z))

def tf_train(X_train, y_train, batch_size=20, n_epoch=1000):
    x = tf.placeholder(tf.float32, [None, D])
    y_ = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.random_normal([D, 1], stddev=1 / np.sqrt(D)))

    # Define loss and optimizer
    z = tf.matmul(x, W)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    # Train
    for epoch in range(n_epoch):
        idx = np.random.choice(len(X_train), batch_size, replace=False)
        _, l = sess.run([train_step, cross_entropy], feed_dict={x: X_train[idx], y_: y_train[idx]})
        if epoch % 100 == 0:
            print('%d loss: ' + str(l))

    return sess.run(W)



data = pd.read_csv('speedbumps.csv')
ind_var = data[['Speed', 'Z', 'z_jolt']]
dep_var = data['speedbump']
dep_var.replace({"no": 0, "yes":1}, inplace=True)

D = ind_var.shape[1]

x_matrix = np.asmatrix(ind_var)
y_matrix = np.transpose(np.asmatrix(dep_var))

weights = tf_train(X_train=x_matrix, y_train=y_matrix)
print(weights)
print ("\n-------------------")

# X_test = 5*np.random.randn(100,D)
# y_test = X_test.dot(w)
# y_test[y_test<=0] = 0
# y_test[y_test>0] = 1

y_inferred = sigmoid(x_matrix.dot(weights)) # Get a probability measure given X
y_inferred[y_inferred>0.5] = 1
y_inferred[y_inferred<=0.5] = 0
#
print(np.sum(dep_var))
print(np.sum(y_inferred))