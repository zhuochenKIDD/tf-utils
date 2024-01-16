import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import get_mutable_dense_hashtable
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.tools import saved_model_utils
from tensorflow.core.framework import types_pb2
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

partition_size = 1
embedding_size = 8

MODEL_PATH = "./saved_model"


def save_table():
    with tf.Session() as sess:
        # embedding table
        table = get_mutable_dense_hashtable(key_dtype=tf.int64,
                                            value_dtype=tf.float32,
                                            shape=tf.TensorShape([embedding_size]),
                                            name="embed_table",
                                            initializer=tf.zeros_initializer(),
                                            shard_num=partition_size)
        # table insert key-value
        key0 = tf.constant([0, 1, 2, 3], dtype=tf.int64, name='key0')
        values0 = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8],
                               [2, 3, 4, 5, 6, 7, 8, 9],
                               [3, 4, 5, 6, 7, 8, 9, 10],
                               [4, 5, 6, 7, 8, 9, 10, 11]], dtype=tf.float32, name='value0')
        insert_op0 = table.insert(keys=key0, values=values0, name='test_insert_1')
        lookup_op0 = table.lookup(keys=key0, name='lookup_op0')
        print(sess.run(insert_op0))

        # query ids
        # ids = tf.constant([0, 2, 0, 1, 4, 5], name='query_ids')
        ids = tf.placeholder(tf.int64, shape=(None, 1), name='query_ids')
        ids_vector = tf.nn.embedding_lookup_hashtable_v2(table, ids)
        ids_vector = tf.identity(ids_vector, name='query_outputs')
        ids_vector_bias = tf.add(ids_vector, 100, name='query_outputs_bias')

        lookup = sess.run(ids_vector, feed_dict={ids: [[0], [1], [2], [3], [4]]})
        print(lookup)

        if True:
            in_sig_def = {'input_ids': utils.build_tensor_info(ids)}
            out_sig_def = {'output_vector': utils.build_tensor_info(ids_vector)}
            sig_def_1 = signature_def_utils.build_signature_def(in_sig_def, out_sig_def)

            in_sig_def = {'output_vector': utils.build_tensor_info(ids_vector)}
            out_sig_def = {'graph_output': utils.build_tensor_info(ids_vector_bias)}
            sig_def_2 = signature_def_utils.build_signature_def(in_sig_def, out_sig_def)

            signature_def_map = {"usr_info": sig_def_1, "graph_output": sig_def_2}
            builder = tf.saved_model.builder.SavedModelBuilder(MODEL_PATH)
            builder.add_meta_graph_and_variables(sess, ['serve'], signature_def_map=signature_def_map)
            builder.save()


def build_input(meta_graph_def, signature_def_key):
    inputs_tensor_info = meta_graph_def.signature_def[signature_def_key].inputs
    inputs_feed_dict = {}
    for tensor_info in inputs_tensor_info.items():
        signature_name, tensor_info = tensor_info[0], tensor_info[1]
        tensor_shape = [d.size for d in tensor_info.tensor_shape.dim]
        if tensor_shape[0] == -1:  # Batch Size
            tensor_shape[0] = 1
        if tensor_info.dtype == types_pb2.DT_INT32:
            np_type = np.int32
        elif tensor_info.dtype == types_pb2.DT_INT64:
            np_type = np.int64
        else:
            np_type = np.float32
        inputs_feed_dict[tensor_info.name] = np.ones(tensor_shape, dtype=np_type)
    return inputs_feed_dict


def get_output(meta_graph_def, signature_def_key):
    out_tensor_info = meta_graph_def.signature_def[signature_def_key].outputs
    return [out_tensor_info[tensor_key].name for tensor_key in out_tensor_info.keys()]


def run_table():
    from tensorflow.python.saved_model import loader
    from tensorflow.python.framework import ops as ops_lib

    meta_graph_def = saved_model_utils.get_meta_graph_def(MODEL_PATH, 'serve')

    usr_input = build_input(meta_graph_def, "usr_info")
    usr_output = get_output(meta_graph_def, "usr_info")
    graph_input = build_input(meta_graph_def, "graph_output")
    graph_output = get_output(meta_graph_def, "graph_output")

    with tf.Session(graph=ops_lib.Graph()) as sess:
        loader.load(sess, ['serve'], MODEL_PATH)

        o = sess.run(usr_output, feed_dict=usr_input)
        print("Input", usr_input)
        print("Output", o)

        o = sess.run(graph_output, feed_dict=graph_input)
        print("Input", graph_input)
        print("Output", o)


if __name__ == "__main__":
    save_table()
    run_table()