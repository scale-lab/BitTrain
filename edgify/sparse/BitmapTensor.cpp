#include <vector>
#include <torch/csrc/autograd/functions/utils.h>
#include <boost/dynamic_bitset.hpp>
#include <torch/extension.h>


class BitmapTensor {
  private:
    std::vector<int64_t> shape;
    std::vector<float_t> values;
    boost::dynamic_bitset<> bitmap;
  public:
    BitmapTensor(torch::Tensor t);
    ~BitmapTensor();
    torch::Tensor get_dense(bool on_gpu);
};

BitmapTensor::BitmapTensor(torch::Tensor t) {
  for (auto s: t.sizes()) {
    shape.push_back(s);
  }
  
  if (t.device().type() == torch::kCPU) {
    auto tf = t.flatten().data<float_t>();
    for (int i = 0; i < t.numel(); i++) {
      if (tf[i] != 0) {
        values.push_back(tf[i]);
        bitmap.push_back(1);
      } else {
        bitmap.push_back(0);
      }
    }
  } else {
    auto tf = t.flatten();
    for (int i = 0; i < t.numel(); i++) {
      auto item = tf[i].item<float_t>();  // this is very slow
      if (item != 0) {
        values.push_back(item);
        bitmap.push_back(1);
      } else {
        bitmap.push_back(0);
      }
    }
  }
  
}
  


BitmapTensor::~BitmapTensor() {}

torch::Tensor BitmapTensor::get_dense(bool on_gpu) {
  std::vector<float> v;
  int j = 0;
  for (auto i = 0; i < (int) bitmap.size(); i++){
    if (bitmap[i] == 1) {
      v.push_back(values[j++]);
    } else {
      v.push_back(0.0);
    }
  }
  
  torch::Tensor t;
  if (on_gpu && torch::cuda::is_available()) {
    t = torch::from_blob(v.data(), shape).clone().to(torch::kCUDA);
  } else {
    t = torch::from_blob(v.data(), shape).clone();
  }

  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BitmapTensor>(m, "BitmapTensor")
        .def(py::init<const torch::Tensor &>())
        .def("get_dense", &BitmapTensor::get_dense);
}

