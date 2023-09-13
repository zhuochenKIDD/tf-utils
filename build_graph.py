import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

import tensorflow as tf


def save_pb(name, graph_def):
    with tf.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())


SIZE = 1024
def build_id(num_input=10, num_output=5):
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[None, SIZE], name="a")

    with tf.device('/cpu:0'):
      a = a + 1.0
      if True:
        a = tf.identity(a)

    b = tf.constant(1.0, shape=[SIZE, SIZE], dtype=tf.float32)
    c = tf.matmul(a, b, name='mat')


    with tf.device('/cpu:0'):
      #for i in range(num_output):
      # c = tf.identity(c)
      c = tf.identity(c, name="output")
  return graph.as_graph_def()


def build_stride_slice():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[None, 8], name="a")
    out = a[:, :]
    out = out[:, tf.newaxis, :]
    return graph.as_graph_def()


def build_div():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[None, 8], name="a")
    # o = tf.div_no_nan(a, 0)
    o = tf.exp(a)
    # o = tf.expm1(a)
    return graph.as_graph_def()

def build_split_concat():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[None, 8], name="a")
    b1, b2, b3 = tf.split(a, 3, axis=0)
    c = tf.concat([b1,b2,b3], axis=1)
    return graph.as_graph_def()


SIZE = 8
def build_tf_trt_graph():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[None, SIZE], name="a")
    #b = tf.constant(1.0, shape=[SIZE, SIZE], dtype=tf.float32)
    #c = tf.matmul(a, b, name='mat')

    bs = tf.shape(a)
    out = tf.fill([bs[0]], 1.0)

    # t = tf.fill([bs[0], SIZE], 1.0)

    # out = tf.multiply(c, t)

    out = tf.identity(out, name="output")
    return graph.as_graph_def()


BS1 = 1
BS2 = 7
def build_split_v():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[1, None, 8, 3], name="a")
    out = tf.split(a, [1,1,1,1,1,1,1,1], -2, name='output')
    for i in range(8):
      o =  tf.identity(out[i], name="o_" + str(i))
    return graph.as_graph_def()

def build_ss():
  with tf.Graph().as_default() as graph:
    a = tf.placeholder(dtype=tf.float32, shape=[1, None, 8, 3], name="a")

    bs2 = tf.shape(a)[1]

    offset = 0
    for i in range(1):
      stride_v = [1,1,1,1]
      start_v = [0, 0, offset, 0]
      offset += 1
      end_v =   [1, bs2, offset, 3]

      print("the", i, "th spec", start_v, end_v)
      o = tf.strided_slice(a, start_v, end_v, stride_v)
      o = tf.identity(o, name="o_" + str(i))
    return graph.as_graph_def()


def test(gdef):
  with tf.Session() as sess:
    tf.import_graph_def(gdef, name='')

    fetchs = ['a:0', 'o_0:0', 'o_5:0', 'o_7:0']
    fetchs = ['a:0', 'o_0:0']
    ret = sess.run(fetchs, feed_dict={'a:0':
      # [[[0.1 + j,  0.2 + j, 0.3 + j] for j in range(BS2)] for i in range(BS1)]
      [[[[k*0.1 + j,  k*0.2 + j, k*0.3 + j] for k in range(8)]  for j in range(BS2)] for i in range(BS1)]
    })
    for n, v in zip(fetchs, ret):
      print(n, v.shape, v)


if __name__ == "__main__":
  with tf.Session() as sess:
    # graph_def = build_split_v()
    graph_def = build_tf_trt_graph()

    save_pb("trt_test.pb", graph_def)
    print(graph_def)
    test(graph_def)

