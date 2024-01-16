import tensorflow as tf

def save_pb(name, graph_def):
    with tf.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())
