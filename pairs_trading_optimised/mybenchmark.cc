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
//#include <immintrin.h>
//#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
#include <arm_neon.h>
#include <array>


using namespace std;

//namespace simd = std::experimental;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "AMD.csv";
    string ms_file = "Intel.csv";

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

        double adjClose = std::stod(row[4]);
        prices.push_back(adjClose);
    }


    return prices;
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 4 == 0, "N should be a multiple of 4 for NEON instructions");
    std::array<float, N> spread;
    size_t spread_index = 0;

    for(size_t i = 0; i < N; ++i) {
        spread[i] = static_cast<float>(stock1_prices[i] - stock2_prices[i]);
    }
    for(size_t i = N; i < stock1_prices.size(); ++i) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        float32x4_t sq_sum_vec = vdupq_n_f32(0.0f);
        for(size_t j = 0; j < N; j += 4) {
            float32x4_t spread_vec = vld1q_f32(&spread[j]);
            sum_vec = vaddq_f32(sum_vec, spread_vec);
            sq_sum_vec = vaddq_f32(sq_sum_vec, vmulq_f32(spread_vec, spread_vec));
        }
        float sum[4], sq_sum[4];
        vst1q_f32(sum, sum_vec);
        vst1q_f32(sq_sum, sq_sum_vec);
        float final_sum = sum[0] + sum[1] + sum[2] + sum[3];
        float final_sq_sum = sq_sum[0] + sq_sum[1] + sq_sum[2] + sq_sum[3];
        float mean = final_sum / N;
        float stddev = std::sqrt(final_sq_sum / N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;
        spread[spread_index] = static_cast<float>(current_spread);
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


