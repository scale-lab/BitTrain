#pragma once

#include <torch/torch.h>

class TBitMap : public torch::Tensor {
private:
    std::vector<int64_t> _tshape;
    std::vector<float_t> _tvalues;
    std::vector<bool> _tbitmap; // http://www.cplusplus.com/reference/vector/vector-bool/
  public:
    TBitMap(torch::Tensor& t);
    void compress();
    void decompress();
};

