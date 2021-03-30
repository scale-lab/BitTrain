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


class Conv2dFunction : public Function<Conv2dFunction> {
 public:
  static torch::Tensor forward(
      AutogradContext *ctx, const torch::Tensor & input, const torch::Tensor & weight,
      torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
    ctx->save_for_backward({input, weight});
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["groups"] = groups;

    std::cout << "deep " << c10::autograd_dispatch_keyset << std::endl;
    std::cout << "hello " << at::impl::variable_excluded_from_dispatch() << std::endl;
    std::cout << "abdo " << input.is_mkldnn() << std::endl;

    return at::native::mkldnn_convolution(
      input, weight, torch::Tensor(),
      padding, stride, dilation, groups);
    // return torch::nn::functional::detail::conv2d(
    //   input, weight, torch::Tensor(),       // bias is currently passed as None
    //   stride, padding, dilation, groups);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = torch::Tensor();

    auto stride = ctx->saved_data["stride"].toIntVector();
    auto padding = ctx->saved_data["padding"].toIntVector();
    auto dilation = ctx->saved_data["dilation"].toIntVector();
    auto groups = ctx->saved_data["groups"].toInt();

    auto grad_output = grad_outputs[0];
    
    // auto grad_input = at::native::mkldnn_convolution_backward_input(input.sizes(), grad_output, weight,  
    //                                                     padding, stride, dilation, groups, false); // bias not defined for now
    // auto grad_weight = at::native::mkldnn_convolution_backward_weights(weight.sizes(), grad_output, input,
    //                                                     padding, stride, dilation, groups, false); // bias not defined for now

    auto grads = at::native::mkldnn_convolution_backward(input, grad_output, weight, 
                                          padding, stride, dilation, groups, {true, true, false});
    
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {std::get<0>(grads), std::get<1>(grads), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    // return {grad_input, std::get<0>(grad_weight), grad_bias, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor conv2d_apply(torch::Tensor & input, const torch::Tensor & weight,
      torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation, int64_t groups) {
  return Conv2dFunction::apply(input, weight, stride, padding, dilation, groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<BitmapTensor>(m, "BitmapTensor")
        .def(py::init<const torch::Tensor &>())
        .def("get_dense", &BitmapTensor::get_dense);
  m.def("conv2d_apply", &conv2d_apply, "Apply Conv2d Function");
}

