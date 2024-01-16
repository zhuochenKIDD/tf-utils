TF_CFLAGS="-I/usr/local/cuda/targets/x86_64-linux/include -I/usr/lib64/python2.7/site-packages/tensorflow_core/include/ -I/usr/lib/python2.7/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0"
TF_LFLAGS="-L/usr/lib64/python2.7/site-packages/tensorflow_core -l:libtensorflow_framework.so.1"
nvcc -std=c++11 -shared fused_tile.cu -o fused_tile.so -DNDEBUG --compiler-options "-fPIC" ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -I/usr/lib/python2.7/site-packages/tensorflow_core/include
