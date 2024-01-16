#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


using namespace tensorflow;

REGISTER_OP("FusedTile")
    .Input("input: float32")     // [1, seq, emb_size]
    .Output("output: float32")   // [bs, seq, emb_size]
    .Attr("tile_bs: int")
    .SetShapeFn(shape_inference::UnknownShape);

class FusedTile : public OpKernel {
 public:
  explicit FusedTile(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tile_bs", &tile_bs));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    auto shape = input_tensor.shape();
    int bs = shape.dim_size(0);
    int seq_len = shape.dim_size(1);
    int emb_size = shape.dim_size(2);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({tile_bs, seq_len, emb_size}), &output));
  }

  int tile_bs;
};

REGISTER_KERNEL_BUILDER(Name("FusedTile").Device(DEVICE_GPU), FusedTile);