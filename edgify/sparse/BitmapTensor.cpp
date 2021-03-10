#include <vector>
#include <torch/extension.h>


// TODO: Define a new data type called BitmapTensor


// TODO: Return bitmap tensor instead
std::vector<int> get_bitmap(torch::Tensor t) {
    // TODO: extract indices of non-zero elements

    // TODO: build bitmap

    // TODO: compress elements to non-zero vector

    auto bm = {5, 7};
    return bm;
}


// TODO: take bitmap tensor instead
torch::Tensor get_tensor(int b) {
    auto t = torch::Tensor();

    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_bitmap", &get_bitmap, "Get bitmap from tensor");
  m.def("get_tensor", &get_tensor, "Get tensor from bitmap");
}

