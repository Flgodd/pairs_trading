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


using namespace std;


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
    static_assert(N % 4 == 0, "N should be a multiple of 4 for SIMD instructions");

    constexpr size_t SpreadSize = 1256;
    alignas(32) std::array<double, SpreadSize * 2> spread;

    __m256d spread_low = _mm256_setzero_pd();
    __m256d spread_high = _mm256_setzero_pd();

    for (size_t i = 0; i < N; i += 4) {
        __m256d stock1 = _mm256_loadu_pd(&stock1_prices[i]);
        __m256d stock2 = _mm256_loadu_pd(&stock2_prices[i]);
        __m256d current_spread = _mm256_sub_pd(stock1, stock2);

        spread_low = _mm256_add_pd(spread_low, current_spread);
        spread_high = _mm256_fmadd_pd(current_spread, current_spread, spread_high);

        _mm256_store_pd(&spread[i * 2], spread_low);
        _mm256_store_pd(&spread[i * 2 + 4], spread_high);
    }

    for (size_t i = N; i < SpreadSize; i += 4) {
        __m256d stock1 = _mm256_loadu_pd(&stock1_prices[i]);
        __m256d stock2 = _mm256_loadu_pd(&stock2_prices[i]);
        __m256d current_spread = _mm256_sub_pd(stock1, stock2);

        __m256d old_stock1 = _mm256_loadu_pd(&stock1_prices[i - N]);
        __m256d old_stock2 = _mm256_loadu_pd(&stock2_prices[i - N]);
        __m256d old_spread = _mm256_sub_pd(old_stock1, old_stock2);

        spread_low = _mm256_sub_pd(_mm256_add_pd(spread_low, current_spread), old_spread);
        spread_high = _mm256_sub_pd(_mm256_fmadd_pd(current_spread, current_spread, spread_high),
                                    _mm256_mul_pd(old_spread, old_spread));

        _mm256_store_pd(&spread[i * 2], spread_low);
        _mm256_store_pd(&spread[i * 2 + 4], spread_high);
    }

    constexpr double inv_n = 1.0 / N;

    for (size_t i = N; i < stock1_prices.size(); ++i) {
        const size_t idx = (i - 1) * 2;
        double mean = spread[idx] * inv_n;
        double stddev = std::sqrt(spread[idx + 1] * inv_n - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
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