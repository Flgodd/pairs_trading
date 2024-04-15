#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <array>
#include <omp.h>


using namespace std;

std::vector<double> stock1_prices;
std::vector<double> stock2_prices;


vector<double> readCSV(const string& filename);



void read_prices() {

    string gs_file = "Intel.csv";
    string ms_file = "AMD.csv";

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
    static_assert(N % 4 == 0, "N should be a multiple of 4 for AVX instructions");

    std::array<double, 19732> spread;
    spread[0] = stock1_prices[0] - stock2_prices[0];
    spread[1] = (stock1_prices[0] - stock2_prices[0]) * (stock1_prices[0] - stock2_prices[0]);

    for (size_t i = 1; i < 19732; i++) {
        const int idx = i * 2;
        double current_spread = stock1_prices[i] - stock2_prices[i];
        spread[idx] = current_spread + spread[idx - 2];
        spread[idx + 1] = (current_spread * current_spread) + spread[idx - 1];
    }

    const int idx = (N - 1) * 2;
    __m256d sum_spread = _mm256_setzero_pd();
    __m256d sum_squared_spread = _mm256_setzero_pd();

    for (size_t i = 0; i <= idx; i += 4) {
        __m256d spread_vec = _mm256_load_pd(&spread[i]);
        sum_spread = _mm256_add_pd(sum_spread, spread_vec);
        sum_squared_spread = _mm256_add_pd(sum_squared_spread, _mm256_mul_pd(spread_vec, spread_vec));
    }

    double mean = _mm256_reduce_add_pd(sum_spread) / N;
    double stddev = std::sqrt(_mm256_reduce_add_pd(sum_squared_spread) / N - mean * mean);

    for (size_t i = N; i < stock1_prices.size(); ++i) {
        const int idx = (i - 1) * 2;
        double current_spread = stock1_prices[i] - stock2_prices[i];

        __m256d spread_vec = _mm256_load_pd(&spread[idx - (N * 2)]);
        __m256d squared_spread_vec = _mm256_load_pd(&spread[idx + 1 - (N * 2)]);

        sum_spread = _mm256_sub_pd(sum_spread, spread_vec);
        sum_squared_spread = _mm256_sub_pd(sum_squared_spread, squared_spread_vec);

        spread_vec = _mm256_set1_pd(current_spread);
        sum_spread = _mm256_add_pd(sum_spread, spread_vec);
        sum_squared_spread = _mm256_add_pd(sum_squared_spread, _mm256_mul_pd(spread_vec, spread_vec));

        mean = _mm256_reduce_add_pd(sum_spread) / N;
        stddev = std::sqrt(_mm256_reduce_add_pd(sum_squared_spread) / N - mean * mean);

        double z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            // Long and Short
        } else if (z_score < -1.0) {
            // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            // Close positions
        } else {
            // No signal
        }
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