#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <iostream>
#include <array>
//#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
//#include <arm_neon.h>
#include <array>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <cuda_runtime.h> // This usually provides basic atomics
#include <atomic>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>

#include "paraForAlt.cuh"

using namespace std;
const int N = 8;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "GS.csv";
    string ms_file = "MS.csv";

    stock1_prices = readCSV(gs_file);
    stock2_prices = readCSV(ms_file);

}

vector<double> readCSV(const string& filename){
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::string> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(value);
        }
        //chenge to 4 for intel and amd and 5 for GS and MS
        double adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    //static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");
    //1256 : 9866
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start);
//    const int NN  = 1256;
    double spread_sum_f[1256];
    double spread_sq_sum_f[1256];
    pairs_trading_strategy_cuda(stock1_prices, stock2_prices);
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    float elapsedTime = 0;
//    cudaEventElapsedTime(&elapsedTime, start, stop);
//    cout<<"pairs_trading_strategy_cuda: "<<elapsedTime<<endl;
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);

    //cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

}

template<size_t N>
void BM_PairsTradingStrategyOptimized(benchmark::State& state) {
    if (stock1_prices.empty() || stock2_prices.empty()) {
        read_prices();
    }
    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);

BENCHMARK_MAIN();

/*int main()
{
    read_prices();
    cout<<stock1_prices.size()<<":"<<stock2_prices.size()<<endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    pairs_trading_strategy_optimized<N>(stock1_prices, stock2_prices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout<<"pairs_trading: "<<elapsedTime<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}*/
