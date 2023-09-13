import tensorflow as tf

from pprint import pprint
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
import numpy as np
import copy


def _read_single_tfrecord(record_file, max_len=1):
  input_strs = []
  cnt = 0
  for serialized_example in tf.python_io.tf_record_iterator(record_file):
    input_strs.append(serialized_example)
    #example = tf.train.Example()
    #example.ParseFromString(serialized_example)
    #feature = example.features.feature

    return serialized_example
    cnt += 1
    if cnt > max_len:
      break
  print("Got input len=", cnt)
  return input_strs


def warmupfile_data(model_file, fname, s_name='predict_ctr', bs=1):
    pb_obj = tf.MetaGraphDef()
    pb_obj = saved_model_pb2.SavedModel()

    file_name = model_file
    with gfile.FastGFile(file_name, 'rb') as f:
        pb_obj.ParseFromString(f.read())
    meta_graphs = pb_obj.meta_graphs
    for meta in meta_graphs:
      signature_def = meta.signature_def
      fun_def = signature_def[s_name]

    fun_inputs = fun_def.inputs

    record = _read_single_tfrecord(fname)
    log = prediction_log_pb2.PredictionLog()
    log.ParseFromString(record)
    input_keys = log.predict_log.request.inputs.keys()
    feed_dict = {}
    for input_tensor_name in fun_inputs:
      #sig_input_key = fun_inputs[input_tensor_name].name
      assert input_tensor_name in input_keys
      v = tf.make_ndarray(log.predict_log.request.inputs[input_tensor_name])
      # v = np.tile(v, (bs, 1))
      # feed_dict[input_tensor_name] = v[:bs]

      feed_dict[input_tensor_name] = v
      # print(input_tensor_name, "'s value shape", feed_dict[input_tensor_name].shape)

    #pprint(feed_dict)
    return feed_dict


r_path = ''
s_method = 'predict_export_outputs'

bs = 1
one_input = warmupfile_data(r_path + "/saved_model.pb", r_path + "/assets.extra/tf_serving_warmup_requests", bs=bs, s_name=s_method)

# pprint(one_input)
