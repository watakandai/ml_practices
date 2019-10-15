#coding: UTF-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def first_example():
    sess = tf.InteractiveSession()
    # tensorboardログ出力ディレクトリ
    log_dir = ".\\log"

    # 計算グラフ定義
    a = tf.constant(1, name='a')
    b = tf.Variable(2, name='b')
    c = tf.Variable(3, name='c')
    op_add = tf.add(a , b)
    op_add2 = tf.add(c , b)
    assign_op = tf.assign(b, op_add)
    assign_op2 = tf.assign(c, op_add2)


    # このコマンドで`op_add`をグラフ上に出力
    tf.summary.scalar('assign_op', assign_op)
    tf.summary.scalar('assign_op2', assign_op2)
    summary_merged = tf.summary.merge_all()

    # グラフを書く
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    # 実行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            sess.run([assign_op, assign_op2])
            summary = sess.run(summary_merged)
            summary_writer.add_summary(summary,i)

    # SummaryWriterクローズ
    summary_writer.close()


if __name__ == "__main__":
    sess = tf.InteractiveSession()

    learning_rate = 0.01
    training_epochs = 500
    batch_size = 128
    display_epoch = 100

    nn_hidden_1 = 256
    nn_hidden_2 = 256
    num_input = 784
    num_output = 10

    # tensorboardログ出力ディレクトリ
    log_dir = ".\\log"

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, num_input], name='x')
    y = tf.placeholder(tf.float32, [None, num_output], name='y_')

    weights = {
        'w1': tf.Variable(tf.random_normal([num_input, nn_hidden_1])),
        'w2': tf.Variable(tf.random_normal([nn_hidden_1, nn_hidden_2])),
        'out': tf.Variable(tf.random_normal([nn_hidden_2, num_output])),
    }
    bias = {
        'b1': tf.Variable(tf.random_normal([nn_hidden_1])),
        'b2': tf.Variable(tf.random_normal([nn_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output])),
    }

    def neural_net(x):
        layer_1 = tf.add(tf.matmul(x, weights['w1']), bias['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), bias['b2'])
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), bias['out'])
        return out_layer

    #W = tf.Variable(tf.zeros([num_input, num_output]))
    #b = tf.Variable(tf.zeros([num_output]))

    with tf.name_scope('Prediction'):
        #pred = tf.nn.softmax(tf.matmul(x, W) + b)
        pred = neural_net(x)

    with tf.name_scope('Loss'):
        #loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar("accuracy", acc)
    summary_merged = tf.summary.merge_all()

    # グラフを書く
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                _, c, summary = sess.run([optimizer, loss, summary_merged],
                                     feed_dict={x: batch_xs, y: batch_ys})
                summary_writer.add_summary(summary, epoch * total_batch + i)
                avg_cost += c / total_batch

            if (epoch+1) % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    #print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
