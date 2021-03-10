import matplotlib.pyplot as plt
import numpy as np

def get_sparse_memory_long(ndim, nse):
    # indices saved as long -> 8 bytes
    return (ndim*8 + 4) * nse

def get_sparse_memory_int(ndim, nse):
    # indices saved as int -> 4 bytes
    return (ndim*4 + 4) * nse

def get_sparse_memory_4b_compressed(ndim, nse):
    # indices are compressed in 4 bytes
    bytes_for_indices = 4
    return (bytes_for_indices + 4) * nse

def get_sparse_memory_bitmap(size, nse):
    # indices are saved in a bitmap that has the same shape
    bytes_for_indices = np.prod(size) / 8
    return bytes_for_indices + (4 * nse)

def get_dense_memory(size):
    return np.prod(size)*4

size = [32, 3, 224, 224]
# size = [32, 16, 224//2, 224//2]
# size = [16, 1024, 512, 512]

number_of_elements = np.prod(size)

x = np.arange(0, number_of_elements, 100)

y0 = len(x)*[get_dense_memory(size)]
y1 = [get_sparse_memory_long(len(size), i) for i in x]
y2 = [get_sparse_memory_int(len(size), i) for i in x]
y3 = [get_sparse_memory_4b_compressed(len(size), i) for i in x]
y4 = [get_sparse_memory_bitmap(size, i) for i in x]

# Convert to percentage
x = [100*i/number_of_elements for i in x]

plt.plot(x, y0, label="Dense")
plt.plot(x, y1, label="Sparse Long Indices")
plt.plot(x, y2, label="Sparse Int Indices")
plt.plot(x, y3, "--" , label="Sparse 4b compressed Indices")
plt.plot(x, y4, "--" , label="Sparse Bitmap Indices")

plt.legend()
plt.xlabel("Percentage of Non Zero Elements")
plt.ylabel("Memory (bytes)")
plt.title(f'size = {size}')
plt.grid()

# plt.show()
plt.savefig(f'sparse_dense{number_of_elements}.png')