import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('../exploratory/campus_speed_bumps.csv')
data = data[['Date', 'Y', 'Z', "Bump"]]
D = 2

idx = np.random.choice(len(data[['Y','Z']]), 20, replace=False)
print (idx)

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
        print(idx)
        for num in idx:
            print("index: %d" % num)
            print(X_train[idx])
        _, l = sess.run([train_step, cross_entropy], feed_dict={x: X_train[idx], y_: y_train[idx]})
        if epoch % 100 == 0:
            print('loss: ' + str(l))

    return sess.run(W)

# print(tf_train(data[['Y','Z']], data['Bump']))