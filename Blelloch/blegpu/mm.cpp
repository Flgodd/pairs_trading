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
#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
#include <arm_neon.h>
#include <array>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>

#include "scan.cuh"
#include "utils.h"

void test(int N) {
    bool canBeBlockscanned = N <= 1024;

    time_t t;
    srand((unsigned)time(&t));
    int *in = new int[N];
    for (int i = 0; i < N; i++) {
        in[i] = rand() % 10;
    }

    printf("%i Elements \n", N);

    // sequential scan on CPU
    int *outHost = new int[N]();
    long time_host = sequential_scan(outHost, in, N);
    printResult("host    ", outHost[N - 1], time_host);

    // full scan
    int *outGPU = new int[N]();
    float time_gpu = scan(outGPU, in, N, false);
    printResult("gpu     ", outGPU[N - 1], time_gpu);

    // full scan with BCAO
    int *outGPU_bcao = new int[N]();
    float time_gpu_bcao = scan(outGPU_bcao, in, N, true);
    printResult("gpu bcao", outGPU_bcao[N - 1], time_gpu_bcao);

    if (canBeBlockscanned) {
        // basic level 1 block scan
        int *out_1block = new int[N]();
        float time_1block = blockscan(out_1block, in, N, false);
        printResult("level 1 ", out_1block[N - 1], time_1block);

        // level 1 block scan with BCAO
        int *out_1block_bcao = new int[N]();
        float time_1block_bcao = blockscan(out_1block_bcao, in, N, true);
        printResult("l1 bcao ", out_1block_bcao[N - 1], time_1block_bcao);

        delete[] out_1block;
        delete[] out_1block_bcao;
    }

    printf("\n");

    delete[] in;
    delete[] outHost;
    delete[] outGPU;
    delete[] outGPU_bcao;
}


using namespace std;

namespace simd = std::experimental;

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

        double adjClose = std::stod(row[5]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, N> spread;
    size_t spread_index = 0;

    test(100);
    for(size_t i = 0; i < N; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    for(size_t i = N; i < stock1_prices.size(); ++i) {
        float64x2_t sum_vec = vdupq_n_f64(0.0);
        float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

        for(size_t j = 0; j < N; j += 2) {
            float64x2_t spread_vec = vld1q_f64(&spread[j]);
            sum_vec = vaddq_f64(sum_vec, spread_vec);
            sq_sum_vec = vaddq_f64(sq_sum_vec, vmulq_f64(spread_vec, spread_vec));
        }


        double sum[2], sq_sum[2];
        vst1q_f64(sum, sum_vec);
        vst1q_f64(sq_sum, sq_sum_vec);
        double final_sum = sum[0] + sum[1];
        double final_sq_sum = sq_sum[0] + sq_sum[1];

        double mean = final_sum / N;
        double stddev = std::sqrt(final_sq_sum / N - mean * mean);

        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;

        spread[spread_index] = current_spread;

        if(z_score > 1.0) {
            // Long and Short
        } else if(z_score < -1.0) {
            // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            // Close positions
        } else {
            // No signal
        }

        spread_index = (spread_index + 1) % N;
        //copm
    }

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

