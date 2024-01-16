import tensorflow as tf
from tensorflow.python.framework import ops
from util import *
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from pprint import pprint

import os
import numpy as np
from tensorflow.python.util import compat
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import constant_op
from tensorflow.python.saved_model import builder as saved_model_builder

def _build_asset_collection(asset_file_name, asset_file_contents,
                            asset_file_tensor_name, asset_subdir=""):
    parent_dir = os.path.join(
        compat.as_bytes("./"), compat.as_bytes(asset_subdir))
    file_io.recursive_create_dir(parent_dir)

    asset_filepath = os.path.join(
        compat.as_bytes(parent_dir), compat.as_bytes(asset_file_name))
    file_io.write_string_to_file(asset_filepath, asset_file_contents)

    asset_file_tensor = constant_op.constant(
        asset_filepath, name=asset_file_tensor_name)
    ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset_file_tensor)
    asset_collection = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
    return asset_collection


def build_saved_model(export_dir):
    with ops.Graph().as_default() as graph:
        with tf.Session() as sess:
            ids = tf.placeholder(tf.float32, shape=(None,), name='input')
            add_v = tf.Variable([2.0], shape=[1], name='weight')
            out = tf.multiply(ids, add_v)
            out = tf.identity(out, name="output")

            #trt_engine_name = tf.constant("trt.engine")
            #tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, trt_engine_name)

            sess.run(tf.global_variables_initializer())
            o = sess.run(out, feed_dict={ids: [1.0, 2.0]})
            print(o)
            pprint(graph._collections)

            save_pb("dbg.pb", sess.graph.as_graph_def())

            in_sig_def = {'input': utils.build_tensor_info(ids)}
            out_sig_def = {'output': utils.build_tensor_info(out)}
            sig_def = signature_def_utils.build_signature_def(in_sig_def, out_sig_def)
            sig_def_map = {"demo": sig_def}

            saver = tf.train.Saver([add_v])
            # v1
            # builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            # v1
            #builder.add_meta_graph_and_variables(sess, ['serve'], signature_def_map=sig_def_map,
            #  saver=saver, assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))

            builder = saved_model_builder._SavedModelBuilder(export_dir)
            asset_list = _build_asset_collection("hello42.txt", "foo bar baz",
                                                 "asset_file_tensor")
            # todo: add CreateTRTResourceHandle
            builder.add_meta_graph_and_variables(sess, ['serve'], signature_def_map=sig_def_map,
              saver=saver, assets_list=asset_list)
            builder.save()


def build_saved_model(save_model_path):
  init_v = np.zeros((32, 32), dtype=np.float32) + 1.0

  with ops.Graph().as_default() as graph:
    with tf.Session() as sess:
      a = tf.placeholder(dtype=tf.float32, shape=[None, 32], name="a")
      # b = tf.Variable(init_v, shape=[32, 32], name="b")
      a = tf.expm1(a)
      b = init_v
      c = tf.matmul(a, b)
      out = tf.identity(c, name="out")

      # sess.run(tf.global_variables_initializer())
      o = sess.run(out, feed_dict={a: init_v})
      print(o)

      save_pb("dbg.pb", sess.graph.as_graph_def())

      in_sig_def = {'input': utils.build_tensor_info(a)}
      out_sig_def = {'output': utils.build_tensor_info(out)}
      sig_def = signature_def_utils.build_signature_def(in_sig_def, out_sig_def, method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
      sig_def_map = {"demo": sig_def}

      builder = tf.saved_model.builder.SavedModelBuilder(save_model_path)
      builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        sig_def_map,
        strip_default_attrs=True)
      builder.save()


def build_saved_model(save_model_path):
  init_v = np.zeros((32, 32), dtype=np.float32) + 1.0

  with ops.Graph().as_default() as graph:
    with tf.Session() as sess:
        from tensorflow.contrib.lookup.lookup_ops import get_mutable_dense_hashtable
        table = get_mutable_dense_hashtable(key_dtype=tf.int64,
                                            value_dtype=tf.float32,
                                            shape=tf.TensorShape([3]),
                                            name="embed_table",
                                            initializer=tf.zeros_initializer(),
                                            shard_num=1)
        
        ids = tf.placeholder(tf.int64, shape=(None, 1), name='query_ids')
        ids_vector = tf.nn.embedding_lookup_hashtable_v2(table, ids)

        save_pb("dbg.pb", sess.graph.as_graph_def())

        in_sig_def = {'input': utils.build_tensor_info(ids)}
        out_sig_def = {'output': utils.build_tensor_info(ids_vector)}
        sig_def = signature_def_utils.build_signature_def(in_sig_def, out_sig_def, 
                                                          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
        sig_def_map = {"demo": sig_def}

        builder = tf.saved_model.builder.SavedModelBuilder(save_model_path)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            sig_def_map,
            strip_default_attrs=True)
        builder.save()
      
      
if __name__ == "__main__":
   import sys
   export_dir = sys.argv[1]
   build_saved_model(export_dir)