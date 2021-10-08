#include <iostream>
#include "tbitmap/tbitmap.h"

TBitMap::TBitMap(torch::Tensor& t) : torch::Tensor(t) {
}

void TBitMap::compress() {

    for (auto s: this->sizes()) {
        _tshape.push_back(s);
    }
    auto tf = this->flatten().data_ptr<float_t>();
    for (int i = 0; i < this->numel(); i++) {
      if (tf[i] != 0) {
        _tvalues.push_back(tf[i]);
        _tbitmap.push_back(1);
      } else {
        _tbitmap.push_back(0);
      }
    }
    this->set_();   // free the underlying memory of tensor
}

void TBitMap::decompress() {
    std::vector<float> v;
    int j = 0;
    for (auto i = 0; i < (int) _tbitmap.size(); i++){
      if (_tbitmap[i] == 1) {
        v.push_back(_tvalues[j++]);
      } else {
        v.push_back(0.0);
      }
    }
    
    this->set_(torch::from_blob(v.data(), _tshape).clone());
    _tvalues.clear();
    _tbitmap.clear();
}

