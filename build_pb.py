import tensorflow as tf

def save_pb(name, graph_def):
    with tf.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())

def build_graph():
  with tf.Graph().as_default() as graph:
    ways = 3
    lst = []
    for i in range(ways):
        input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        o = tf.multiply(input_tensor, 3)
        lst.append(o)
    output_tensor = tf.concat(lst, axis=1)
  return graph


if __name__ == "__main__":
  with tf.Session() as sess:
    graph = build_graph()
    save_pb("sess_graph.pb", graph.as_graph_def())
