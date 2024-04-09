#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <array>
#include <chrono>
#include <cstddef>



using namespace std;

struct StockPrices {
    vector<double> stock1;
    vector<double> stock2;
};

StockPrices readCSV(const string& filename1, const string& filename2);

void read_prices(StockPrices& prices) {
    string gs_file = "GS.csv";
    string ms_file = "MS.csv";
    prices = readCSV(gs_file, ms_file);
}

StockPrices readCSV(const string& filename1, const string& filename2) {
    StockPrices prices;
    prices.stock1.reserve(1000);
    prices.stock2.reserve(1000);

    std::ifstream file1(filename1);
    std::ifstream file2(filename2);
    std::string line;

    std::getline(file1, line);
    std::getline(file2, line);

    while (std::getline(file1, line) && std::getline(file2, line)) {
        std::stringstream ss1(line);
        std::stringstream ss2(line);
        std::string value;
        std::vector<std::string> row1, row2;

        while (std::getline(ss1, value, ',')) {
            row1.push_back(value);
        }
        while (std::getline(ss2, value, ',')) {
            row2.push_back(value);
        }

        double adjClose1 = std::stod(row1[5]);
        double adjClose2 = std::stod(row2[5]);
        prices.stock1.push_back(adjClose1);
        prices.stock2.push_back(adjClose2);
    }

    return prices;
}

template<size_t N, size_t UnrollFactor>
struct LoopUnroll {
    static void computeSpread(std::array<double, N>& spread, const vector<double>& stock1_prices,
                              const vector<double>& stock2_prices, size_t startIndex,
                              double& sum, double& sq_sum) {
        spread[startIndex] = stock1_prices[startIndex] - stock2_prices[startIndex];
        spread[startIndex + 1] = stock1_prices[startIndex + 1] - stock2_prices[startIndex + 1];
        sum += spread[startIndex] + spread[startIndex + 1];
        sq_sum += (spread[startIndex] * spread[startIndex]) + (spread[startIndex + 1] * spread[startIndex + 1]);
        LoopUnroll<N, UnrollFactor - 2>::computeSpread(spread, stock1_prices, stock2_prices, startIndex + 2, sum, sq_sum);
    }
};

template<size_t N>
struct LoopUnroll<N, 0> {
    static void computeSpread(std::array<double, N>& spread, const vector<double>& stock1_prices,
                              const vector<double>& stock2_prices, size_t startIndex,
                              double& sum, double& sq_sum) {
        // Base case, do nothing
    }
};


template<size_t N>
void pairs_trading_strategy_optimized(const StockPrices& prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");
    std::array<double, N> spread;
    size_t spread_index = 0;
    double sum = 0.0;
    double sq_sum = 0.0;

    LoopUnroll<N, N>::computeSpread(spread, prices.stock1, prices.stock2, 0, sum, sq_sum);

    uint32_t d = N;
    uint64_t c = UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1;

    for (size_t i = N; i < prices.stock1.size(); ++i) {
        double mean = sum / N;
        double stddev = std::sqrt(sq_sum / N - mean * mean);
        double current_spread = prices.stock1[i] - prices.stock2[i];
        double z_score = (current_spread - mean) / stddev;
        double old_value = spread[spread_index];
        spread[spread_index] = current_spread;

        if (z_score > 1.0) {
            // Long and Short
        } else if (z_score < -1.0) {
            // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            // Close positions
        } else {
            // No signal
        }

        sum += -old_value + current_spread;
        sq_sum += -(old_value * old_value) + (current_spread * current_spread);

        uint64_t lowbits = c * (spread_index + 1);
        spread_index = ((__uint128_t)lowbits * d) >> 64;
    }
}


template<size_t N>
void BM_PairsTradingStrategyOptimized(benchmark::State& state) {
    StockPrices prices;
    read_prices(prices);

    for (auto _ : state) {
        pairs_trading_strategy_optimized<N>(prices);
    }
}

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 8);
BENCHMARK_MAIN();
