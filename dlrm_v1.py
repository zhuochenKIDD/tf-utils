# Copyright 2023 The RECom Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import string
import os
import sys
import random
import shutil
from multiprocessing import Process

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

tf.disable_resource_variables()
tf.disable_eager_execution()
tf.random.set_random_seed(0)
np.random.seed(0)


small_embedding_table_rows = 100
mid_embedding_table_rows = 10000
large_embedding_table_rows = 1 << 23
embedding_dim = 8
large_embedding_dim = 32
input_rows = 256 # bs?
input_cols = 10


def make_num_input():
    return np.random.randint(100, size=input_rows)


def make_str_input():
    ALPHABET = list(string.ascii_letters + string.digits)
    return [''.join(list(np.random.choice(ALPHABET, 8))) for _ in range(input_rows)]


def make_sparse_str_input():
    ALPHABET = list(string.ascii_letters + string.digits)
    return [';'.join([''.join(list(np.random.choice(ALPHABET, 8)))
                     for _ in range(random.randint(1, input_cols))])
            for _ in range(input_rows)]


def make_categ_hashbucket_int(feature_name, features, columns, feed_dict, input_sig):
    input_data = make_num_input()
    placeholder = tf.placeholder(
        dtype=tf.int32, name=feature_name, shape=[None])
    fc = tf.feature_column.categorical_column_with_hash_bucket(
        feature_name,
        small_embedding_table_rows,
        dtype=tf.int32)
    fc_emb = tf.feature_column.embedding_column(
        fc, embedding_dim, combiner='mean')
    columns.append(fc_emb)

    features[feature_name] = placeholder
    feed_dict[feature_name + ':0'] = input_data
    input_sig[feature_name] = placeholder


def make_categ_hashbucket(feature_name, features, columns, feed_dict, input_sig):
    input_data = make_str_input()
    placeholder = tf.placeholder(
        dtype=tf.string, name=feature_name, shape=[None])
    fc = tf.feature_column.categorical_column_with_hash_bucket(
        feature_name,
        mid_embedding_table_rows,
        dtype=tf.string)
    fc_emb = tf.feature_column.embedding_column(
        fc, embedding_dim, combiner='mean')
    columns.append(fc_emb)

    features[feature_name] = placeholder
    feed_dict[feature_name + ':0'] = input_data
    input_sig[feature_name] = placeholder


def make_categ_hashbucket_sparse(feature_name, features, columns, feed_dict, input_sig):
    input_data = make_sparse_str_input()
    placeholder = tf.placeholder(
        dtype=tf.string, name=feature_name, shape=[None])
    fc = tf.feature_column.categorical_column_with_hash_bucket(
        feature_name,
        mid_embedding_table_rows,
        dtype=tf.string)
    fc_emb = tf.feature_column.embedding_column(
        fc, embedding_dim, combiner='sum')
    columns.append(fc_emb)

    features[feature_name] = tf.strings.split(placeholder, sep=';')
    feed_dict[feature_name + ':0'] = input_data
    input_sig[feature_name] = placeholder


def make_large_categ_hashbucket_sparse(feature_name, features, columns, feed_dict, input_sig):
    input_data = make_sparse_str_input()
    placeholder = tf.placeholder(
        dtype=tf.string, name=feature_name, shape=[None])
    fc = tf.feature_column.categorical_column_with_hash_bucket(
        feature_name,
        large_embedding_table_rows,
        dtype=tf.string)
    fc_emb = tf.feature_column.embedding_column(
        fc, large_embedding_dim, combiner='sum')
    columns.append(fc_emb)

    features[feature_name] = tf.strings.split(placeholder, sep=';')
    feed_dict[feature_name + ':0'] = input_data
    input_sig[feature_name] = placeholder


def make_bucketize(feature_name,
                   features,
                   columns,
                   feed_dict,
                   input_sig):

    input_data = make_num_input()
    # shape = [256], value = [0, 99)
    boundary = [i for i in range(0, small_embedding_table_rows * 5, 5)]
    # shape = [100], value = [0,5 ... 490,495]
    placeholder = tf.placeholder(
        dtype=tf.float32, name=feature_name, shape=[None])
    """
        feature_name = 'f1'
        NumericColumn(key='f1', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)
    """
    age = tf.feature_column.numeric_column(feature_name)
    """
        BucketizedColumn(
            source_column=NumericColumn(key='f1', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), 
            boundaries=(0, 5, 10))
    """
    fc = tf.feature_column.bucketized_column(age, boundaries=boundary)
    """
        _EmbeddingColumn(
            categorical_column=BucketizedColumn(
                source_column=NumericColumn(key='f1', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), 
                boundaries=(0, 5, 10)), 
            dimension=8, 
            combiner='mean', 
            layer_creator=<function _creator at 0x7f2318c12de8>, 
            ckpt_to_load_from=None, 
            tensor_name_in_ckpt=None, 
            max_norm=None, 
            trainable=True, 
            with_hashtable=False, 
            threshold=0)

    """
    fc_emb = tf.feature_column.embedding_column(
        fc,
        embedding_dim,  # 8
        combiner='mean')
    
    columns.append(fc_emb)

    features[feature_name] = placeholder
    feed_dict[feature_name + ':0'] = input_data
    input_sig[feature_name] = placeholder


def create_dlrm(bucketize_num,  # 800
                categ_hashbucket_int_num,  # 50
                categ_hashbucket_num,  # 50
                categ_hashbucket_sparse_num,  # 15
                categ_large_hashbucket_sparse_num, # 5
                dense_num,  # 32
                bot_units,  # [32]
                top_units,  # [1024, 1024, 128]
                model_name):
    # For convenience, we use the same configurations for columns belonging to
    # the same type. Note that in real industrial models, columns with the same
    # topological structures can also have varied specifications (e.g. #rows,
    # embedding dimension, bucket boundaries, etc.)
    fc_gen = {}
    for i in range(bucketize_num):
        fc_gen['bucketize' + str(i)] = make_bucketize
    for i in range(categ_hashbucket_int_num):
        fc_gen['categ_hashbucket_int' + str(i)] = make_categ_hashbucket_int
    for i in range(categ_hashbucket_num):
        fc_gen['categ_hashbucket' + str(i)] = make_categ_hashbucket
    for i in range(categ_hashbucket_sparse_num):
        fc_gen['categ_hashbucket_sparse' + str(i)] = make_categ_hashbucket_sparse
    for i in range(categ_large_hashbucket_sparse_num):
        fc_gen['categ_large_hashbucket_sparse' + str(i)] = make_large_categ_hashbucket_sparse

    features = {}
    columns = []
    feed_dict = {}   # 输入
    input_sig = {}   # 签名
    for name, generator in fc_gen.items():
        generator(name, features, columns, feed_dict, input_sig)
    sparse_inputs = tf.feature_column.input_layer(features, columns)


    # dense输入
    dense_placeholder = tf.placeholder(dtype=tf.float32, name='dense', shape=[None, dense_num])
    feed_dict['dense:0'] = np.random.randint(100, size=[input_rows, dense_num]).astype(np.float32)
    input_sig['dense'] = dense_placeholder

    dense_inputs = dense_placeholder
    for i, hidden_units in enumerate(bot_units):
        dense_inputs = tf.layers.dense(dense_inputs, units=hidden_units,
                                       activation=tf.nn.relu, name= 'bot' + str(i))
        

    # sparse + dense输入
    mlp_input = tf.concat([dense_inputs, sparse_inputs], 1)

    # 3层MLP
    for i, hidden_units in enumerate(top_units):
        mlp_input = tf.layers.dense(mlp_input,
                                    units=hidden_units,
                                    activation=tf.nn.relu,
                                    name= 'top' + str(i))

    # 最终的输出    
    logits = tf.layers.dense(mlp_input, units=1, activation=None)
    probability = tf.math.sigmoid(logits)
    output = tf.round(probability)

    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0,
                            device_count={'GPU': 0})
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(output, feed_dict)
        shutil.rmtree(model_name, ignore_errors=True)
        tf.saved_model.simple_save(
            sess, model_name, input_sig, {'output': output})


if __name__ == '__main__':
    # Algorithm developers tend to generate many numeric statistic features and
    # then use bucketization to transform them into sparse features. We observe
    # these features can account for the vast majority of all features in many
    # industrial models.
    for args in [
        (1, 2, 3, 4, 5, 6, 
         [32], 
         [1024, 1024, 128], 
         'Test')
    ]:
        print('Creating ', args[-1])
        p = Process(target=create_dlrm, args=args)
        p.start()
        p.join()
    print('Note that for real industrial models, the configurations '
          '(e.g. #rows, embedding dim, bucket boundaries) can vary '
          'among different columns')