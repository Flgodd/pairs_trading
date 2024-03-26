#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
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
#include <thread>


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

    std::array<double, 1256> spread;
    std::array<double, 1256> spread_sum;
    std::array<double, 1256> spread_sq_sum;
    std::array<double, 1256*2> spread_sum_comb;
    vector<int> check(4, 0);
    int numThreads = 4;

    int chunk_size = stock1_prices.size() / numThreads;

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunk_size;
        int end = (i == numThreads - 1) ? stock1_prices.size() : (i + 1) * chunk_size;

        // Create threads directly with inline lambda function
        threads.emplace_back([start, end, &stock1_prices, &stock2_prices, &spread]() {
            for (int i = start; i < end; ++i) {
                spread[i] = stock1_prices[i] - stock2_prices[i];
            }
        });
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto worker = [&](size_t start_index, size_t end_index) {
        for (size_t i = start_index; i < end_index; i++) {
            const int idx = i * 2;
            float64x2_t sum_vec = vdupq_n_f64(0.0);
            float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

            for (size_t j = i - N; j < i; j += 2) {
                float64x2_t spread_vec = vld1q_f64(&spread[j]);
                sum_vec = vaddq_f64(sum_vec, spread_vec);
                sq_sum_vec = vaddq_f64(sq_sum_vec, vmulq_f64(spread_vec, spread_vec));
            }


            double sum[2], sq_sum[2];
            vst1q_f64(sum, sum_vec);

            vst1q_f64(sq_sum, sq_sum_vec);
            double final_sum = sum[0] + sum[1];
            double final_sq_sum = sq_sum[0] + sq_sum[1];

            spread_sum_comb[idx] = final_sum;
            spread_sum_comb[idx + 1] = final_sq_sum;
        }
    };

    chunk_size = (spread_sum.size()-N) / numThreads;
    threads.clear();

    for (int i = 0; i < numThreads; ++i) {
        int start = i * chunk_size + N;
        int end = (i == numThreads - 1) ? stock1_prices.size() : (i + 1) * chunk_size;

        threads.emplace_back(worker, start, end+N);
    }

    // Wait for threads to complete
    for (auto& thread : threads) {
        thread.join();
    }





    for (size_t i = N; i < stock1_prices.size(); ++i) {
        const int idx = i*2;
        //const int idx = (i-1)*2;
        double mean = (spread_sum_comb[idx])/ N;
        double stddev = std::sqrt((spread_sum_comb[idx+1])/ N - mean * mean);
        //double current_spread = spread[i];
        double z_score = (spread[i] - mean) / stddev;


        if (z_score > 1.0) {
            check[0]++;  // Long and Short
        } else if (z_score < -1.0) {
            check[1]++;  // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            check[2]++;  // Close positions
        } else {
            check[3]++;  // No signal
        }

    }
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