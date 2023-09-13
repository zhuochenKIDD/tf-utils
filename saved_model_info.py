import tensorflow as tf
import os
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import tensor_shape_pb2

import numpy as np
from pprint import pprint
from config_model import r_path


model_dir = "/data1/chenzhuo/ad-rec/identity_no_fuseconst"
file_name = model_dir + "/saved_model.pb"


def save_pb(name, graph_def):
    with tf.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())

def pn(n):
  #print(n)
  #print(n.op)
  print(n.name)
  print(n.input)


def reserve_nodes(node_map, name):
    ret = set()
    nodes_to_add = set()
    nodes_to_add.add(name)
    while len(nodes_to_add) > 0:
      cur_node = nodes_to_add.pop()
      ret.add(cur_node)
      for tensor_node in node_map[cur_node].input:
        nodes_to_add.add(tensor_node.split(":")[0])
    return ret

def build_node_map(graph_def):
    ret = {}
    for n in graph_def.node:
      ret[n.name] = n
    return ret

def fit(graph_def):
  for n in graph_def.node:
    if n.op in ['LookupTableImport', 'LookupTableFind', 'LookupTableExport', 'LookupTableInsert', 'MutableDenseHashTable']:
      if '_class' in n.attr:
        del n.attr['_class']

def get_const_data(node_def):
  tensor_value = node_def.attr['value']
  real_shape = [d.size for d in tensor_value.tensor.tensor_shape.dim]
  tensor_data = np.frombuffer(
    tensor_value.tensor.tensor_content,
    dtypes.DType(tensor_value.tensor.dtype).as_numpy_dtype()
  ).reshape(real_shape)
  return tensor_data


def dfs_find_ph(node_map, find_node):
  inputs_to = set()
  inputs_to.add(find_node)
  while len(inputs_to) > 0:
    visit = inputs_to.pop()
    visit = node_map[visit]

    if visit.op == 'Placeholder':
      print("Find Node", find_node, "PH:", visit.attr['shape'])
      return
    else:
      for i_t in visit.input:
        i_node = i_t.split(":")[0].split("^")[-1]
        inputs_to.add(i_node)


def op_stat(graph_def):
  op_2_cnt = {}
  node_num = 0
  for n in graph_def.node:
    if n.op not in op_2_cnt.keys():
      op_2_cnt[n.op] = 0
    op_2_cnt[n.op] += 1
    node_num += 1

  print("Node Num In Graph", node_num)
  import operator
  from pprint import pprint
  sorted_op = sorted(op_2_cnt.items(), key=operator.itemgetter(1))
  pprint(sorted_op)


pb_obj = tf.MetaGraphDef()
pb_obj = saved_model_pb2.SavedModel()

with gfile.FastGFile(file_name, 'rb') as f:
  pb_obj.ParseFromString(f.read())

meta_graphs = pb_obj.meta_graphs

def small_graph(graph_def, scope_name):
  name_2_node = {}
  new_graph_def = tf.GraphDef()
  to_add = set()
  added_node = set()
  cnt = 0

  for node in graph_def.node:
    name_2_node[node.name] = node
    if scope_name in node.name:
      added_node.add(node.name)
      new_node = new_graph_def.node.add()
      new_node.CopyFrom(node)
      cnt += 1
      for i_tensor in new_node.input:
        if scope_name not in i_tensor:
          if i_tensor.split(':')[0] not in added_node:
            to_add.add(i_tensor.split(':')[0])

  print("Add ", cnt , " node to graph")
  save_pb(scope_name.split("/")[-1] + ".pb", new_graph_def)
  # print(new_graph_def)
  # return new_graph_def

  while len(to_add) > 0:
    node = to_add.pop()
    node = name_2_node[node.split('^')[-1]]
    if node.name in added_node:
      continue
    added_node.add(node.name)
    new_node = new_graph_def.node.add()
    new_node.CopyFrom(node)
    cnt += 1
    for i_tensor in new_node.input:
      if i_tensor.split(':')[0] not in added_node:
        to_add.add(i_tensor.split(':')[0])
  print("Add ", cnt , " node to graph")
  save_pb(scope_name.split("/")[-1] + ".pb", new_graph_def)
  return new_graph_def

for meta in meta_graphs:
 graph_def = meta.graph_def
 saver_def = meta.saver_def
 collection_def = meta.collection_def
 #print(collection_def)
 object_graph_def = meta.object_graph_def
 op_stat(graph_def)

 if not True:
   with tf.Session() as sess:
     graph_def = del_xla_attr(graph_def)
     graph_def = small_graph(graph_def, 'sku_name_Seq_Embedding/concat_7')
     tf.import_graph_def(graph_def, name='')
     op_stat(graph_def)
     #save_pb("graph_def.pb", graph_def)

     model_dir = './tb'
     train_writer = tf.summary.FileWriter(model_dir)
     train_writer.add_graph(sess.graph)
