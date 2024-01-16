
import onnx
import operator
from pprint import pprint


onnx_file = "/Users/zhuochen/Documents/github/tensorflow-onnx/0_0.onnx"

onnx_model = onnx.load(onnx_file)

def op_stat(graph):
    op_type_2_cnt = {}
    ns_2_cnt = {}
    for n in graph.node:
        if n.op_type not in op_type_2_cnt:
            op_type_2_cnt[n.op_type] = 0
        op_type_2_cnt[n.op_type] += 1

        ns = n.name.split("/")[0]
        if ns not in ns_2_cnt:
            ns_2_cnt[ns] = 0
        ns_2_cnt[ns] += 1
    
    sorted_op = sorted(op_type_2_cnt.items(), key=operator.itemgetter(1))
    pprint(sorted_op)
    # pprint(sorted(ns_2_cnt.items(), key=operator.itemgetter(1)))



def build_node_map(graph):
    ret = {}
    for n in graph.node:
      ret[n.name] = n
    return ret

def initializer_info(initializer):
    name_2_info = {}
    for i in initializer:
        # print("name=", i.name, i.dims)
        name_2_info[i.name] = i
    return name_2_info


def inspect_graph(graph):
    print("graph name:", graph.name, ", node len=", len(graph.node), 
          ", initializer len=", len(graph.initializer))

    nm = build_node_map(graph)
    name_2_info = initializer_info(graph.initializer)
    op_stat(graph)

    for n in graph.node:
        # if n.op_type in ['Gemm']:
        if n.op_type in ['MatMul']:
            if n.input[1] not in name_2_info:
                print("fk", n.name)
                continue
            B = name_2_info[n.input[1]]
            print(n.name, "NK=", B.dims)



# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
def inspect_model(onnx_file):
    onnx_model = onnx.load(onnx_file)
    print("ONNX IR Ver:", onnx_model.ir_version)
    # print(onnx_model.opset_import)

    # print(onnx_model.functions) # []
    # print(onnx_model.metadata_props) # []

    graph = onnx_model.graph
    inspect_graph(graph)

if __name__ == "__main__":

    inspect_model(onnx_file)