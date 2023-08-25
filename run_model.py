import os
import time

import tensorflow as tf
from tensorflow.contrib import stat_summarizer
from tensorflow.python.saved_model import loader
from tensorflow.python.tools import saved_model_utils
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from pprint import pprint

skip_var_lst = [
]

skip_var_lst = []

def op_stat(graph_def):
    print("Node Op Info================")
    op_2_cnt = {}
    node_num = 0
    for n in graph_def.node:
        if n.op not in op_2_cnt:
            op_2_cnt[n.op] = 0
        op_2_cnt[n.op] = op_2_cnt[n.op] + 1
        node_num += 1

    format_str = "Node Num In Graph: " + str(node_num)
    print(format_str)
    import operator
    from pprint import pprint
    sorted_op = sorted(op_2_cnt.items(), key=operator.itemgetter(1))

    format_str += "\n" + "\n".join([str(i) for i in sorted_op])
    return format_str


def prune_input(graph_def, orig_feed_input):
    new_feed_input = {}
    for n in graph_def.node:
        if 'Placeholder' == n.op:
          input_key = n.name + ":0"
          if input_key in orig_feed_input:
            new_feed_input[input_key] = orig_feed_input[input_key]
    return new_feed_input



def run_saved_model(saved_model_dir, tag_set, signature_def_key,
                    input_tensor_key_feed_dict,
                    enable_trace=False,
                    warm_up=20, repeats=40):
    # Get a list of output tensor names.
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                          tag_set)

    inputs_feed_dict = {}
    # Re-create feed_dict based on input tensor name instead of key as session.run
    # uses tensor name.
    inputs_tensor_info = meta_graph_def.signature_def[signature_def_key].inputs
    for input_key_name in input_tensor_key_feed_dict.keys():
        if input_key_name not in inputs_tensor_info:
            print('%s is not a valid input key ' % input_key_name)
        else:
          inputs_feed_dict[inputs_tensor_info[input_key_name].name] = input_tensor_key_feed_dict[input_key_name]

    outputs_tensor_info = meta_graph_def.signature_def[signature_def_key].outputs
    output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
    output_tensor_names_sorted = [
        outputs_tensor_info[tensor_key].name
        for tensor_key in output_tensor_keys_sorted
    ]

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
        gpu_options=gpu_options,
        graph_options=graph_options,
        experimental=tf.ConfigProto.Experimental(
            executor_type="DEFAULT",
            use_numa_affinity=True,
            disable_thread_spinning=True,
            optimize_for_static_graph=True,
        ),
    )
    config=tf.ConfigProto(log_device_placement=False)

    from tensorflow.python.framework import ops as ops_lib
    with tf.Session(graph=ops_lib.Graph(), config=config) as sess:
        loader.load(sess, tag_set.split(','), saved_model_dir)
        print("Loader Load Done")
        print("============= output_tensor_names_sorted ======= ")
        print(output_tensor_names_sorted)
        # for aedl it cores
        inputs_feed_dict = prune_input(sess.graph_def, inputs_feed_dict)
        #inputs_feed_dict['input/tf_example:0'] = [""]
        #pprint(inputs_feed_dict)

        acc = 0
        min_rt = 10000000000000

        for s in range(1):
          outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)

        print("Warmup Ends")

        repeats = 1
        # repeats = 100

        if enable_trace:
          repeats = 10

        dur_lst = []

        for k, v in inputs_feed_dict.items():
          break

        repeats = 3
        print("FK sess.run input k=", k, "v shape=", v.shape)
        print("FK output_tensor_names_sorted ====>")
        # output_tensor_names_sorted = ['concat:0', 'Sigmoid:0', '0_0:0']
        print(output_tensor_names_sorted)
        for s in range(repeats):
            metadata = tf.RunMetadata()

            start = time.time()
            if not enable_trace:
                outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict, run_metadata=metadata)

            else:
                print("Enable Tracing")
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
                #options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
                outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict,
                                   options=options, run_metadata=metadata)

            dur = metadata.step_stats.step_run_elapsed_micros
            print("dur======", dur)
            if dur == 0:
              dur = 1000 * (time.time() - start)  # ms
            else:
              dur = 0.001 * dur
            dur_lst.append(dur)
            acc += dur
            if dur < min_rt:
                min_rt = dur

        from pprint import pprint
        # pprint(outputs)
        # return

        if enable_trace:
            stat = stat_summarizer.NewStatSummarizer(b"unused")
            stat.ProcessStepStatsStr(metadata.step_stats.SerializeToString())
            stat.PrintStepStats()
            from tensorflow.python.client import timeline
            tl = timeline.Timeline(metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        avg_rt = round(acc / repeats, 3)
        min_rt = round(min_rt, 3)
        print("Avg RT: ", avg_rt, " (ms), Min RT: ", min_rt, " (ms)")

        import numpy as np
        np.set_printoptions(precision=10)
        for i, output in enumerate(outputs):
            if i < len(output_tensor_keys_sorted):
              output_tensor_key = output_tensor_keys_sorted[i]
            else:
              output_tensor_key = output_tensor_names_sorted[i]
            print('Result for output key %s:\n%s' % (output_tensor_key, output.shape))
            pprint(output)

if __name__ == "__main__":
    from config_model import r_path,one_input,s_method
    run_saved_model(r_path, 'serve', s_method, one_input, enable_trace=True)