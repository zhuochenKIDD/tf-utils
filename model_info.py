import tensorflow as tf
import os
import sys

from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import tensor_shape_pb2


import numpy as np
from pprint import pprint


def load_pb(filename):
  with tf.io.gfile.GFile(filename,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    return graph_def

def save_pb(name, graph_def):
    with tf.io.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())

def pn(n):
  #print(n)
  #print(n.op)
  print(n.name)
  print(n.input)


def build_node_map(graph_def):
    ret = {}
    for n in graph_def.node:
      ret[n.name] = n
    return ret


def get_const_data(node_def):
  tensor_value = node_def.attr['value']
  real_shape = [d.size for d in tensor_value.tensor.tensor_shape.dim]
  tensor_data = np.frombuffer(
    tensor_value.tensor.tensor_content,
    dtypes.DType(tensor_value.tensor.dtype).as_numpy_dtype()
  ).reshape(real_shape)
  return tensor_data

def op_stat(graph_def):
  op_2_cnt = {}
  node_num = 0
  for n in graph_def.node:
    node_num += 1
    if n.name in ['init']:
      print(n)
    if n.op not in op_2_cnt.keys():
      op_2_cnt[n.op] = 0
    op_2_cnt[n.op] += 1

  print("Node Num In Graph", node_num)
  import operator
  from pprint import pprint
  sorted_op = sorted(op_2_cnt.items(), key=operator.itemgetter(1))
  pprint(sorted_op)


def input_nodes(node_map, graph_def, node_name):
    to_add = set()
    added_node = set()
    to_add.add(node_name)

    cnt = 0
    while len(to_add) > 0:
        node = node_map[to_add.pop()]
        added_node.add(node.name)
        cnt += 1
        for i_tensor in node.input:
          if i_tensor.startswith('^'):
              continue
          i_node = i_tensor.split(":")[0]
          to_add.add(i_node)
    return added_node


def clean_const_node(gdef):
  for n in gdef.node:
    if n.op == "Const":
      tensor_value = n.attr['value'].tensor
      tensor_value.ClearField("tensor_content")

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise

  name = sys.argv[1]
  graph_def = load_pb(name)
  op_stat(graph_def)
  print(graph_def)

