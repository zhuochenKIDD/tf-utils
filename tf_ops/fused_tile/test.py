import tensorflow as tf
from pprint import pprint

bs = 1
seq_len = 3
emb_size = 8

tile_bs = 200
test_data = [[[s * 0.1 for _ in range(emb_size) ] for s in range(seq_len)] for _ in range(bs)]

def test_fused_tile():
    fused_tile = tf.load_op_library("./fused_tile.so")

    input_data = test_data

    with tf.Session() as sess:
        out = fused_tile.FusedTile(input_data, tile_bs=tile_bs)
    ret = sess.run(out)
    pprint(ret)


def test_tf():
  with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        a = tf.placeholder(dtype=tf.float32, shape=[bs, seq_len, emb_size], name="a")
        tile_out = tf.tile(a, [tile_bs, 1, 1], name='tile_out')
        broadcast_out = tf.broadcast_to(a, [tile_bs, seq_len, emb_size], name='broadcast_out')

        ret = sess.run(tile_out, feed_dict={a: test_data})
        print(ret)

  return graph.as_graph_def()


if __name__ == "__main__":
    test_tf()