# -*- coding: utf-8 -*- 
import json
import sys
import os


class InputInfo:
    def __init__(self, input_node_idx, input_node_output_idx=0):
        super().__init__()
        self.input_node_idx = input_node_idx
        self.input_node_output_idx = input_node_output_idx

    def __str__(self):
        return "InputIdx=" + str(self.input_idx)

    def __repr__(self):
        return self.__str__()


class NodeEntry:
    def __init__(self, node_id, index, version):
        super().__init__()
        self.node_id = node_id
        self.index = index
        self.version = version


class TVMOpParam:
    def __init__(self, func_name, num_inputs, num_outputs, 
                 flattern_data, attrs):
        super().__init__()
        self.func_name = func_name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.flattern_data = flattern_data
        self.attrs = attrs
    
    def __str__(self):
        return "TVMOpParam = [func_name=" + self.func_name + ",num_inputs=" + self.num_inputs + ",num_outputs=" + self.num_outputs
    
    def __repr__(self):
        return self.__str__()


class TvmOp:
    def __init__(self, idx, op_type, name, inputs, attrs):
        super().__init__()
        self.idx = idx
        self.op_type = op_type  # tvm_op or null
        self.name = name
        self.inputs = inputs
        self.tvm_op_param = None
        # self.input_nodes = []
        if op_type == 'tvm_op':
            # for input_info in inputs:
            #     self.input_nodes.append(InputInfo(input_node_idx=input_info[0], input_node_output_idx=input_info[1]))
            self.tvm_op_param = TVMOpParam(attrs['func_name'], attrs['num_inputs'], attrs['num_outputs'], attrs['flatten_data'], None)
    def __str__(self):
        return "Idx=" + str(self.idx)  \
            + ",op_type=" + self.op_type \
            + ",op_name=" + self.name \
            + ",tvm_params=" + str(self.tvm_op_param) \
            + ",input" + str(self.inputs)
    
    def __repr__(self):
        return self.__str__()
    
def entry_id(node_id, index, node_row_ptr_):
    return node_row_ptr_[node_id] + index

def parse_graph_json(json_file):
    with open(json_file, 'r') as f:
      json_str = f.read()
      tvm_graph = json.loads(json_str)

    # 全部的TVM节点数
    all_tvm_nodes = tvm_graph["nodes"]
    input_nodes = tvm_graph["arg_nodes"]
    output_nodes = tvm_graph["heads"] 
    attrs = tvm_graph["attrs"]
    node_row_ptr = tvm_graph["node_row_ptr"]
    attr_shape_lst,  dltype_lst, storage_id_lst = attrs['shape'][-1], attrs['dltype'][-1], attrs['storage_id'][-1]

    # 这里实际是Tensor的信息。有些Tensor没有用到，比如某个OP有多个输出，只用到了1个
    # data entry is all the data the functions needs to access
    data_entry_shapes = []
    num_node_entries = node_row_ptr[-1]
    for i in range(num_node_entries):
        data_entry_shapes.append(attr_shape_lst[i])
    
    # node --> (shape, type, storage_id)
    node_2_attrs = {}
    for i in range(len(attr_shape_lst)):
        # 代表某一个entry的shape，entry作为function的argv和retv
      node_2_attrs[i] = (attr_shape_lst[i], dltype_lst[i], storage_id_lst[i])        
    
    index_2_tvm_op = {}
    node_idx_map = {} # <node_idx -> TvmOp>
    for i in range(len(all_tvm_nodes)):
        tvm_op = all_tvm_nodes[i]
        op_type = tvm_op['op']
        op_name = tvm_op['name']
        if op_type == 'tvm_op':
            node = TvmOp(i, op_type, op_name, tvm_op['inputs'], tvm_op['attrs'])
        else:
            node = TvmOp(i, op_type, op_name, [], None)
        node_idx_map[i] = node
        # if 'attrs' in tvm_op:
        #   tvm_op = TvmOp(i, op_type, op_name, tvm_op['inputs'], tvm_op['attrs'])
        # else:
        #   tvm_op = TvmOp(i, op_type, op_name, tvm_op['inputs'])
        #  index_2_tvm_op[i] = tvm_op
    # pprint(tvm_op_type_2_cnt)
    # pprint(index_2_tvm_op)
    # pprint(node_2_attrs)
    output_node_idx = []
    for output in output_nodes:
        output_node_idx.append(output[0])
    return node_idx_map, node_row_ptr, attr_shape_lst, output_node_idx

def simplify_name(name):
    return name
    if name.startswith("fused"):
        ops = name.split("_")
        return "_".join(ops[2:])
    return name

def json_to_graphviz(node_idx_map, node_row_ptr, attr_shape_lst, output_nodes):
    import graphviz
    dot = graphviz.Digraph(comment='TVM Graph')
    skip_param_nodes = []
    for node_idx, tvm_node in node_idx_map.items():
        if tvm_node.op_type == "tvm_op":
            # pass
            # if node_info.idx in output_nodes:
            #     node_attr = node_2_attrs[node_info.idx]
            #     node_shape = node_attr[0]
            #     dot.node(str(node_id), node_info.name + "\nOutputShape:" + str(node_shape), style='filled', fillcolor='olive', shape='box')
            # else:
            #     # colors: https://www.graphviz.org/doc/info/colors.html
            #     dot.node(str(node_id), node_info.name, style='filled', fillcolor='bisque')
            # dot.node(tvm_node.name, '', style='filled', fillcolor='bisque')
            if node_idx in output_nodes:
                eid = entry_id(node_idx, 0, node_row_ptr)
                shape_info = attr_shape_lst[eid]
                output_name = "TVMOutput_" + str(node_idx)                
                dot.node(output_name, style='filled', fillcolor='olive', shape='box')
                dot.edge(simplify_name(tvm_node.name), output_name, label=str(shape_info))
            # colors: https://www.graphviz.org/doc/info/colors.html
            dot.node(simplify_name(tvm_node.name), style='filled', fillcolor='bisque')
        else:
            if "TVMInputPH" not in tvm_node.name:
                skip_param_nodes.append(node_idx)
                continue
            else:
                # dot.node(tvm_node.name, '', style='filled', fillcolor='tomato', shape='invtriangle')
                dot.node(simplify_name(tvm_node.name), style='filled', fillcolor='tomato', shape='invtriangle')
    for node_idx, tvm_node in node_idx_map.items():
        for input_node_info in tvm_node.inputs:
            node_id, index = input_node_info[0], input_node_info[1]
            if node_id not in skip_param_nodes:
                input_node = node_idx_map[node_id]
                input_eid = entry_id(node_id, index, node_row_ptr)
                shape_info = attr_shape_lst[input_eid]
                dot.edge(simplify_name(input_node.name), simplify_name(tvm_node.name), label=str(shape_info))
    return dot
    dot.render('tvm_graph.gv').replace('\\', '/')

if __name__ == "__main__":
    if len(sys.argv) is not 3:
        print("Usage: python tvm_graph_to_pb.py ${TVM GraphRuntime Json File}  ${GraphDef Output Dir}")
        print("Example: python tvm_graph_to_pb.py /workdir/tf_tvm_files/0_0.json .")
    else:
        tvm_graph_json = sys.argv[1]
        output_dir = sys.argv[2]
        node_idx_map, node_row_ptr, attr_shape_lst, output_node_idx = parse_graph_json(tvm_graph_json)
        dot_graph = json_to_graphviz(node_idx_map, node_row_ptr,  attr_shape_lst, output_node_idx)
        dot_graph.render(os.path.join(output_dir, 'tvm_graph.gv')).replace('\\', '/')