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
#include <thread>
#include <vector>
#include <mutex>


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

//std::mutex spread_mutex; // Global mutex to protect access to shared resources

template<size_t N>
void process_chunk(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices, size_t start, size_t end, std::array<double, N>& spread, std::mutex& spread_mutex) {
    for(size_t i = start; i < end; ++i) {
        if (i < N) continue; // Skip the initial filling of the spread array
        float64x2_t sum_vec = vdupq_n_f64(0.0);
        float64x2_t sq_sum_vec = vdupq_n_f64(0.0);

        for(size_t j = 0; j < N; j += 2) {
            float64x2_t spread_vec;
            // Protect reading from spread with mutex
            {
                std::lock_guard<std::mutex> guard(spread_mutex);
                spread_vec = vld1q_f64(&spread[j]);
            }
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

        // Protect writing to spread with mutex
        std::lock_guard<std::mutex> guard(spread_mutex);
        spread[i % N] = current_spread;
        // Logic to handle z_score (e.g., long/short positions) goes here

    }
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, N> spread;
    std::mutex spread_mutex;
    size_t spread_index = 0;
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Initialize spread array as before

    size_t chunk_size = stock1_prices.size() / num_threads;
    for(size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i + 1) * chunk_size;
        if (i == num_threads - 1) end = stock1_prices.size(); // Make sure the last chunk includes the end
        threads.emplace_back(process_chunk<N>, std::ref(stock1_prices), std::ref(stock2_prices), start, end, std::ref(spread), std::ref(spread_mutex));
    }

    for(auto& thread : threads) {
        thread.join();
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
