#include <vector>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/torch.h>
#include <boost/dynamic_bitset.hpp>
#include <torch/extension.h>

using namespace torch::autograd;

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
  std::cout << values.size() << "/" << t.numel() << " (" << (float) values.size() / (float) t.numel() * 100.0 << "%)" <<  std::endl;
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


class LinearFunction : public Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
      AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};

torch::Tensor apply(torch::Tensor input, torch::Tensor weight) {
  return LinearFunction::apply(input, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BitmapTensor>(m, "BitmapTensor")
        .def(py::init<const torch::Tensor &>())
        .def("get_dense", &BitmapTensor::get_dense);
  m.def("apply", &apply, "Apply Linear Function");
}

