# Bitmap Tensor

## Download libtorch

First, download libtorch from this link: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). Choose the right platform and OS, since the build is platform-dependent.

## Build project

Use `cmake` to build the example project. 
```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
cmake --build .
```

The example files shows the compression and decompression operations of the activations (*not the layer parameters*).
You can experiment with different number of elements and different sparsity levels using:

```
./example num_elements %non-zero
```

