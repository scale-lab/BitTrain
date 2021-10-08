#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <regex>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cmath>
#include "tbitmap/tbitmap.h"

using namespace std;

int mem_uss() {
  std::ifstream smaps_file("/proc/self/smaps");
  std::string line;
  std::regex expr("Private.*: *(\\d+).*");
  
  int uss {0};
  while(std::getline(smaps_file, line)){
    std::smatch sm;
    if(std::regex_match(line, sm, expr, regex_constants::match_default)) {
      uss += std::stoi(sm[1]);
    }
  }
  return uss * 1024; // return results in bytes
}


int main(int argc, char **argv) {
  if(argc < 2) {
    std::cout << "Run ./example 'num_elements %non-zero'\n";
    return 0;
  }
  
  int batch_size = 16;
  auto n = batch_size * stoi(argv[1]);

  double non_zero = stof(argv[2]);
  
  auto _ones = torch::ones({int(batch_size * non_zero), n});
  auto _zeros = torch::zeros({int(batch_size * (1 - non_zero)), n});
  
  int base = mem_uss() / (1024.0 * 1024.0);     // memory before the target tensor

  auto tensor = torch::cat({_ones, _zeros}, 0);
  
  auto x = TBitMap(tensor);
  std::cout << n << "," << (mem_uss() / (1024.0 * 1024.0)) - base;

  x.compress();
  std::cout << "," << (mem_uss() / (1024.0 * 1024.0)) - base << std::endl;
}

