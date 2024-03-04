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


using namespace std;

namespace simd = std::experimental;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;

constexpr size_t STOCK_SIZE = 1256; //for loop unrolling example gotta keep const otherwise runtime.


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

template<size_t Index, size_t N>
struct InnerLoopUnroll {
    static void execute(const std::array<double, N>& spread, float64x2_t& sum_vec, float64x2_t& sq_sum_vec) {
        if constexpr (Index < N) {
            float64x2_t spread_vec = vld1q_f64(&spread[Index]);
            sum_vec = vaddq_f64(sum_vec, spread_vec);
            sq_sum_vec = vaddq_f64(sq_sum_vec, vmulq_f64(spread_vec, spread_vec));

            InnerLoopUnroll<Index + 2, N>::execute(spread, sum_vec, sq_sum_vec); // Increment by 2
        }
    }
};

// Outer loop unrolling template
template<size_t I, size_t N>
struct OuterLoopUnrollFirstSeg {
    static void execute(vector<int>& check, size_t& spread_index, const vector<double>& stock1_prices, const vector<double>& stock2_prices, array<double, N>& spread) {
        if constexpr (I < 1000) {
            const size_t i = I;
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

            InnerLoopUnroll<0, N>::execute(spread, sum_vec, sq_sum_vec);

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
                check[0]++;
            } else if(z_score < -1.0) {
                // Short and Long
                check[1]++;
            } else if (std::abs(z_score) < 0.8) {
                // Close positions
                check[2]++;
            } else {
                // No signal
                check[3]++;
            }
            spread_index = (spread_index + 1) % N;

            OuterLoopUnrollFirstSeg<I + 1, N>::execute(check, spread_index, stock1_prices, stock2_prices, spread);
        }
    }
};

template<size_t I, size_t N, size_t startIdx>
struct OuterLoopUnrollSecondSeg {
    static void execute(vector<int>& check, size_t& spread_index, const vector<double>& stock1_prices, const vector<double>& stock2_prices, array<double, N>& spread) {
        if constexpr (I < startIdx + 256) {
            const size_t i = I;
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

            InnerLoopUnroll<0, N>::execute(spread, sum_vec, sq_sum_vec);

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
                check[0]++;
            } else if(z_score < -1.0) {
                // Short and Long
                check[1]++;
            } else if (std::abs(z_score) < 0.8) {
                // Close positions
                check[2]++;
            } else {
                // No signal
                check[3]++;
            }

            spread_index = (spread_index + 1) % N;

            OuterLoopUnrollSecondSeg<I + 1, N, startIdx>::execute(check, spread_index, stock1_prices, stock2_prices, spread);
        }
    }
};


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, N> spread;
    size_t spread_index = 0;

    for(size_t i = 0; i < N; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    vector<int> check(4, 0);

    OuterLoopUnrollFirstSeg<N, N>::execute(check, spread_index, stock1_prices, stock2_prices, spread);
    OuterLoopUnrollSecondSeg<1000, N, 1000>::execute(check, spread_index, stock1_prices, stock2_prices, spread);

    cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

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
