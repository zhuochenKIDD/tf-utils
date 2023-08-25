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


bs = 2
def build_batch_matmul():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[bs, 2, 3], name="a")
    b = tf.constant([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]] for i in range(bs)], tf.float32)
    c = tf.matmul(a, b, name="ret")
  return graph.as_graph_def()


def build_graph(ph_num=2, dense_width=1, dense_depth=1):
  bs = 2
  with tf.Graph().as_default() as graph:
    """
    ph = []
    for i in range(4):
      ph.append(tf.placeholder(dtype=tf.float32, shape=[bs, 3 + i]))
    output_tensor = tf.concat(ph, axis=1, name='output_concat')
    """
    a = tf.placeholder(dtype=tf.float32, shape=[bs, 3], name="a")


    for l in [1, 2]:
      k =  tf.constant([[0.1*l,0.2*l,0.3],
                        [0.5*l,0.6,0.7],
                        [0.9*l,1.0,1.1]], tf.float32, name="k" + str(l))
      a =  tf.matmul(a, k)
      a = a + [[ l*1.0*(i+10)*(j+2) for j in [1,2,3] ] for i in range(bs)]
      a = a - [[ l*2.0*(i+10)*(j+2) for j in [1,2,3] ] for i in range(bs)]
      a = a * [[ l*3.0*(i+10)*(j+2) for j in [1,2,3] ] for i in range(bs)]
      a = a / [[ l*4.0*(i+10)*(j+2) for j in [1,2,3] ] for i in range(bs)]

  return graph.as_graph_def()


def eval_tf(graph_def):
  with tf.Session() as sess:
    tf.import_graph_def(graph_def, name="")
    ret = sess.run("ret:0", feed_dict={"a:0": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] for i in range(bs)]})
    print(ret)
    print(ret.shape)


def build_mean():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[bs, 2, 3], name="a")
    c = tf.reduce_mean(a, axis=2, keepdims=True, name="ret")
  return graph.as_graph_def()

if __name__ == "__main__":
  with tf.Session() as sess:
    # graph_def = build_batch_matmul()
    graph_def = build_mean(True)
    save_pb("mean.pb", graph_def)
    eval_tf(graph_def)
