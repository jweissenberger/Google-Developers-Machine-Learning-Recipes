# This code is run after the Inception v3 model is retrained to classify kinds of flowers from a repository of
# images of different flowers

import os, sys
import tensorflow as tf

# changes the path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# takes in the image information
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# loads the label file, takes of the carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# removes graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # inputs the image data to the graph
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # the first prediction
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

    # sort the labels from highest to least confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))