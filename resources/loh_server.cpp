#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <fstream>

// Needed for HTTPS, implies the need for openssl, may be omitted if HTTP suffices
// #define CPPHTTPLIB_OPENSSL_SUPPORT

const std::size_t OUTPUT_SIZE = 123;

#include "umbridge.h"

class ExaSeisModel : public umbridge::Model {
public:

  ExaSeisModel()
   : umbridge::Model("forward")
  {}

   // Define input and output dimensions of model (here we have a single vector of length 1 for input; same for output)
  std::vector<std::size_t> GetInputSizes(const json& config_json) const override {
    return {3};
  }

  std::vector<std::size_t> GetOutputSizes(const json& config_json) const override {
    std::vector<std::size_t> output;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      output.push_back(4);
    }
    return output;
  }

  std::vector<std::vector<double>> Evaluate(const std::vector<std::vector<double>>& inputs, json config) override {
    char const* thread_num_cstr = std::getenv("OMP_NUM_THREADS");
    std::string thread_num = "0";
    if ( thread_num_cstr == NULL ) {
      thread_num = "1";
    } else {
      thread_num = thread_num_cstr;
    }

    std::ofstream inputFile("/app/input.txt", std::ios::trunc);
    if (inputFile.is_open()) {
        inputFile << inputs[0][0] << std::endl;
        inputFile << inputs[0][1] << std::endl;
        inputFile << inputs[0][2] << std::endl;
        inputFile.close();
    }
    std::string cmd = "cd /Peano/applications/exahype2/exaseis/Cartesian && OMP_NUM_THREADS=" + thread_num + " ./LOH_level_1_order_1_legendre_precision_double";
    std::cout << "cmd: " << cmd << std::endl;
    int status = system(cmd.c_str());
    if (status != 0) {
      std::cout << "Exahype exit status " << status << std::endl;
      return {{-1, -1, -1}};
    }
    std::ifstream outputFile("/app/tracers/Cartesian_level_1_order_1_legendre_precision_double-rank-0.csv");
    std::string line;
    std::vector<double> t;
    std::vector<double> v0;
    std::vector<double> v1;
    std::vector<double> v2;
    for (int i = 0; i <= OUTPUT_SIZE && std::getline(outputFile, line); ++i) {
      if (i == 0) continue;
      std::stringstream ss(line);
      std::string cell;
      int cellCount = 0;
      while (std::getline(ss, cell, ',')) {
        cellCount++;
        if (cellCount == 3) t.push_back(std::stod(cell));
        else if (cellCount == 7) v0.push_back(std::stod(cell));
        else if (cellCount == 8) v1.push_back(std::stod(cell));
        else if (cellCount == 9) v2.push_back(std::stod(cell));
      }
    }
    outputFile.close();
    std::vector<std::vector<double>> output;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
      output.push_back({t[i], v0[i], v1[i], v2[i]});
    }
    return output;
  }

  // Specify that our model supports evaluation. Jacobian support etc. may be indicated similarly.
  bool SupportsEvaluate() override {
    return true;
  }
};

int main(){

  // Read environment variables for configuration
  char const* port_cstr = std::getenv("PORT");
  int port = 0;
  if ( port_cstr == NULL ) {
    std::cout << "Environment variable PORT not set! Using port 4243 as default." << std::endl;
    port = 4242;
  } else {
    port = atoi(port_cstr);
  }

  ExaSeisModel model;
  umbridge::serveModels({&model}, "0.0.0.0", port);

  return 0;
}