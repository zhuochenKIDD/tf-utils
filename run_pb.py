# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import sys
from tensorflow.contrib import stat_summarizer
import os
import time
from pprint import pprint

trace = False


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


def load_pb(filename):
  with tf.gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    return graph_def

def save_pb(name, graph_def):
    with tf.gfile.FastGFile(name,'wb') as f:
        f.write(graph_def.SerializeToString())

def get_feed_dict(graph_def, graph, batch):
    batch = int(batch)
    bs = batch
    feed_dict = {}

    if not False:
      cnt = 1
      for node in graph_def.node:
        if node.op != 'Placeholder':
            continue

        tensor = graph.get_tensor_by_name(node.name + ":0")
        dtype = tensor.dtype
        shape = tensor.shape.as_list()
        for i, dim in enumerate(shape):
            if dim == None:
                shape[i] = batch
        if dtype != tf.string:
            data = np.zeros(tuple(shape), dtype.as_numpy_dtype) + cnt
            cnt += 1
        else:
            data = [['1_41_42_43_41_44_55_66_54_35_'*2 + '1']]*batch
        feed_dict[tensor] = data
    return feed_dict


def main(name, batch, outputs):
    graph_def = load_pb(name)
    
    run_options = tf.RunOptions(
        trace_level=tf.RunOptions.FULL_TRACE, 
        report_tensor_allocations_upon_oom=True,
        experimental=tf.RunOptions.Experimental(
            use_run_handler_pool=True
        ),
    )
    run_metadata = tf.RunMetadata()

    opt_options = tf.OptimizerOptions(
        do_common_subexpression_elimination=True,
        do_constant_folding=True,
        do_function_inlining=True,
    )

    graph_options = tf.GraphOptions(
        optimizer_options=opt_options,
        build_cost_model=not True,
        infer_shapes=True,
        enable_bfloat16_sendrecv=True, # grpc session
    )

    graph_options.rewrite_options.disable_meta_optimizer = True

    from tensorflow.core.protobuf import rewriter_config_pb2
    graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF


    gpu_options = tf.GPUOptions(
        allow_growth = not True,
        experimental=tf.GPUOptions.Experimental(
            num_dev_to_dev_copy_streams=4,
            timestamped_allocator=False,
        )
    )

    config = tf.ConfigProto(
        intra_op_parallelism_threads=3,
        inter_op_parallelism_threads=2,
        gpu_options=gpu_options,
        graph_options=graph_options,
        experimental=tf.ConfigProto.Experimental(
            executor_type="DEFAULT",
            use_numa_affinity=True,
            disable_thread_spinning=True,
            optimize_for_static_graph=True,
        ),
    )

    outputs = outputs.split(',')
    tensors = []
    for node in outputs:
        if ':' not in node:
            node = node + ':0'
        tensors.append(node)

    acc = 0
    min_rt = 10000000000000
    repeats = 100
    repeats = 3

    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            tf.import_graph_def(graph_def, name='')
            op_stat(graph_def)

            feed_dict = get_feed_dict(graph_def, graph, batch)
            print("========== feed_dict ===========")
            print(feed_dict)

            for i in range(2):
              result = sess.run(tensors, feed_dict=feed_dict)

            #raise

            if not trace:
                for i in range(repeats):
                    start = time.time()
                    result = sess.run(tensors, feed_dict = feed_dict)
                    dur = 1000 * (time.time() - start)
                    acc += dur
                    if dur < min_rt:
                      min_rt = dur
            else:
                for i in range(repeats):
                    result = sess.run(tensors, feed_dict=feed_dict,
                                      options=run_options,
                                      run_metadata=run_metadata)

        avg_rt = round(acc / repeats, 3)
        min_rt = round(min_rt, 3)
        print("Avg RT: ", avg_rt, " (ms), Min RT: ", min_rt, " (ms)")

    if trace:
        pprint(run_metadata)

        stat = stat_summarizer.NewStatSummarizer(b"unused")
        stat.ProcessStepStatsStr(run_metadata.step_stats.SerializeToString())
        stat.PrintStepStats()


        from tensorflow.python.client import timeline
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        return

    for r, name in zip(result, outputs):
        print(name, r.shape, r) 

if __name__ == '__main__':
    file_name = sys.argv[1]
    batch = int(sys.argv[2])
    outputs = sys.argv[3]
    main(file_name, batch, outputs)