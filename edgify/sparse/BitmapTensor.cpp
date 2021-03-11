#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <torch/extension.h>


class BitmapTensor {
  private:
    float_t* values;
    int8_t* shape;
    boost::dynamic_bitset<> bitmap;
  public:
    BitmapTensor(torch::Tensor t);
    ~BitmapTensor();
    torch::Tensor get_dense();
};

BitmapTensor::BitmapTensor(torch::Tensor t) {
  // TODO: convert dense to our format
  std::cout << "I'm a construtor with torch input" << std::endl;
}

torch::Tensor BitmapTensor::get_dense() {
  // TODO: reconstruct dense from our format
  auto t = torch::Tensor();
  std::cout << "I will return a dense tensor" << std::endl;

  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BitmapTensor>(m, "BitmapTensor")
        .def(py::init<const torch::Tensor &>())
        .def("get_dense", &BitmapTensor::get_dense);
}

