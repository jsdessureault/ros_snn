import numpy as np
import tensorflow as tf
import csv
import sys

# Inspire de: https://github.com/nethsix/gentle_tensorflow/blob/master/code/linear_regression_multi_feature_using_mini_batch_with_tensorboard.py

filename = 'ROS_SNN_results.csv'

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 1000
epoch = 1000
learn_rate = 0.001
log_file = "./log/"
feature_nb = 8
data_nb = 6

# Features
i_sensory = 4.0
i_inter = 4.0
i_layer = 1.0
i_motor = 2.0
i_synapse = 8.0
i_instance = 1.0
i_input = 1.0
i_benchmark = 1.2


def input_prediction_parameters():
    global i_sensory,i_inter,i_layer,i_motor,i_synapse,i_instance,i_input,i_benchmark
    print ("Predicting SNN cycle time (in ms):")

    old_sensory = i_sensory
    i_sensory = input("Number of sensory neurons (default: "+str(i_sensory)+"): ")
    if (i_sensory == ""): i_sensory = old_sensory

    old_inter = i_inter
    i_inter = input("Number of inter neurons (default: "+str(i_inter)+"): ")
    if (i_inter == ""): i_inter = old_inter

    old_layer = i_layer
    i_layer = input("Number of inter layer (default: "+str(i_layer)+"): ")
    if (i_layer == ""): i_layer = old_layer

    old_motor = i_motor
    i_motor = input("Number of motor neurons (default: "+str(i_motor)+"): ")
    if (i_motor == ""): i_motor = old_motor

    old_synapse = i_synapse
    i_synapse = input("Number of synapses (default: "+str(i_synapse)+"): ")
    if (i_synapse == ""): i_synapse = old_synapse

    old_instance = i_instance
    i_instance = input("Number of instances (default: "+str(i_instance)+"): ")
    if (i_instance == ""): i_instance = old_instance

    old_input = i_input
    i_input = input("Sensory input usage % (default: "+str(i_input)+"): ")
    if (i_input == ""): i_input = old_input

    old_benchmark = i_benchmark
    i_benchmark = input("CPU Benchmark (default: "+str(i_benchmark)+"): ")
    if (i_benchmark == ""): i_benchmark = old_benchmark


# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [data_nb, feature_nb], name="x")
W = tf.Variable(tf.zeros([feature_nb,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
with tf.name_scope("Wx_b") as scope:
  product = tf.matmul(x,W)
  y = product + b
y_ = tf.placeholder(tf.float32, [data_nb, 1])

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
  cost_sum = tf.summary.scalar("cost", cost)

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_xs = []
all_ys = []

input_prediction_parameters()

# Assign the data read from file
with open(filename, 'r') as f:
    reader = csv.DictReader(f, delimiter=",", quotechar="|")
    try:
        for row in reader:
            #print (row)
            all_xs.append([float(row['sensories']), float(row['inters']), float(row['layers']), float(row['motors']), float(row['synapses']), float(row['instances']), float(row['sensory_input']), float(row['benchmark'])])
            all_ys.append(float(row['cycle_time']))
    except csv.Error as e:
        sys.exit('Error: file %s, line %d: %s' % (filename, reader.line_num, e))

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])

sess = tf.Session()
merged = tf.summary.merge_all()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(epoch):
  # Process with chunks of size batch_sixe
  if datapoint_size == batch_size:
    batch_start_idx = 0
  elif datapoint_size < batch_size:
    raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
  else:
    batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
  batch_end_idx = batch_start_idx + batch_size
  # Extract chunks from orignal array. 
  batch_xs = all_xs[batch_start_idx:batch_end_idx]
  batch_ys = all_ys[batch_start_idx:batch_end_idx]
  xs = np.array(batch_xs)
  ys = np.array(batch_ys)
  all_feed = { x: all_xs, y_: all_ys }
  feed = { x: xs, y_: ys }
  sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %s" % sess.run(W))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=all_feed))

predict_X = np.array([i_sensory,i_inter,i_layer,i_motor,i_synapse,i_instance,i_input,i_benchmark], dtype=np.float32).reshape((1,feature_nb))
#predict_X = np.array([3.0,4.0,1.0,4.0,8.0,1.0,1.0,1.2], dtype=np.float32).reshape((1,feature_nb))
print("Predicting:")
print (predict_X)
#Normalize
#predict_X = predict_X / np.linalg.norm(predict_X)
predict_Y = tf.add(tf.matmul(predict_X, W),b)
print ("Predicted cycle time: ")
print (sess.run(predict_Y))


