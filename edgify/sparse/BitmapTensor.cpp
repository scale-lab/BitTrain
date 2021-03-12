#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <torch/extension.h>


class BitmapTensor {
  private:
    torch::IntArrayRef shape;
    std::vector<float_t> values;
    boost::dynamic_bitset<> bitmap;
  public:
    BitmapTensor(torch::Tensor t);
    ~BitmapTensor();
    torch::Tensor get_dense();
};

BitmapTensor::BitmapTensor(torch::Tensor t) {
  shape = t.sizes();
  
  std::vector<float_t> v(t.flatten().data_ptr<float_t>(), t.flatten().data_ptr<float_t>() + t.flatten().numel());
  for (auto el: v) {
    if (el != 0) {
      values.push_back(el);
      bitmap.push_back(1);
    } else {
      bitmap.push_back(0);
    }
  }
}

BitmapTensor::~BitmapTensor() {}

torch::Tensor BitmapTensor::get_dense() {
  std::vector<float> v;
  int j = 0;
  for (auto i = 0; i < (int) bitmap.size(); i++){
    if (bitmap[i] == 1) {
      v.push_back(values[j++]);
    } else {
      v.push_back(0.0);
    }
  }

  torch::Tensor t = torch::from_blob(v.data(), shape).clone();
  
  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BitmapTensor>(m, "BitmapTensor")
        .def(py::init<const torch::Tensor &>())
        .def("get_dense", &BitmapTensor::get_dense);
}

