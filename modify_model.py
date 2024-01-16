import tensorflow as tf
import os
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.util import compat
from tensorflow.core.framework import tensor_shape_pb2

from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saved_model_pb2


from config_model import r_path
file_name = r_path + "/saved_model.pb"

def op_stat(graph_def):
  op_2_cnt = {}
  node_num = 0
  for n in graph_def.node:
    if n.op not in op_2_cnt:
      op_2_cnt[n.op] = 0

    op_2_cnt[n.op] = op_2_cnt[n.op] + 1
    node_num += 1

  print("Node Num In Graph", node_num)
  import operator
  from pprint import pprint
  sorted_op = sorted(op_2_cnt.items(), key=operator.itemgetter(1))
  pprint(sorted_op)


def set_device(graph_def):
    for n in graph_def.node:
      if n.op == 'CuFuncRun':
        continue
      if n.op == 'Const' and "/p" in n.name:
        continue
      n.device = '/device:CPU:0'

def remove_dropout(graph_def):
  for n in graph_def.node:
    if 'gate_output/Add' == n.name:
      n.input[0] = 'gate_output/MatMul:0'
    if 'gate_output/Add_1' == n.name:
      n.input[0] = 'gate_output/MatMul_1:0'


def clear_device(graph_def):
  input_nodes = {}
  idx_to_shape = {}
  for n in graph_def.node:
    n.device = ''
    for i in n.input:
      if 'IteratorGetNext' in i:
        input_nodes[n.name] = i
    if 'IteratorGetNext' == n.name:
      print(n)
      num = len(n.attr['_output_shapes'].list.shape)
      i = 0
      while i < num:
        idx_to_shape[i] = (n.attr['_output_shapes'].list.shape[i], n.attr['output_types'].list.type[i])
        i+=1

  from pprint import pprint
  pprint(input_nodes)
  print(idx_to_shape)


def replace_ph_name(graph_def, ph_node_2_id):
  for n in graph_def.node:
    if n.op == 'Placeholder':
      if n.name in ph_node_2_id.keys():
        n.name = ph_node_2_id[n.name]
    else:
      for idx, i_tensor in enumerate(n.input):
        i_node = i_tensor.split(":")[0]
        if i_node in ph_node_2_id.keys():
          n.input[idx] = ph_node_2_id[i_node] + ":0"


def disable_import(graph_def):
  for n in graph_def.node:
    if 'save/restore_shard' in n.name:
      idx = -1
      for i in range(len(n.input)):
        if 'LookupTableImport' in n.input[i]:
          idx = i
          break
      if idx > 0:
        print(n.input[idx:])
        del n.input[idx:]
      print(n)


def build_node_map(graph_def):
    node_map = {} # string to node
    for node in graph_def.node:
        node_map[node.name] = node
    return node_map

def remove_id_cufun(graph_def):
  nm = build_node_map(graph_def)
  for n in graph_def.node:
    if n.op == 'Identity' and 'batch' in n.name:
      print(n.name)
      print(n.input)

    if n.op == 'CuFuncRun':
      id_node = n.input[0]
      i_node = nm[id_node].input[0]

      if 'batch' not in id_node:
        continue
      if id_node in ['batch_5_0_1', 'batch_2_0_1']:
        # print(nm[id_node])
        continue
      n.input[0] = i_node

pb_obj = tf.MetaGraphDef()
pb_obj = saved_model_pb2.SavedModel()

with gfile.FastGFile(file_name, 'rb') as f:
  pb_obj.ParseFromString(f.read())

meta_graphs = pb_obj.meta_graphs

for meta in meta_graphs:
  graph_def = meta.graph_def
  meta.ClearField('graph_def')

  remove_id_cufun(graph_def)
  # disable_import(graph_def)
  meta.graph_def.CopyFrom(graph_def)


if True:
  with gfile.FastGFile(file_name, 'wb') as f:
    f.write(pb_obj.SerializeToString())
