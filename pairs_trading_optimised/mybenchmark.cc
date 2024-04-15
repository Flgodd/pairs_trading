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
    std::vector<double> spread(19732);
    spread[0] = stock1_prices.at(0) - stock2_prices.at(0);
    spread[1] = (stock1_prices.at(0) - stock2_prices.at(0)) * (stock1_prices.at(0) - stock2_prices.at(0));

#pragma omp simd
    for (size_t i = 1; i < 19732; i++) {
        const size_t idx = i * 2;
        double current_spread = stock1_prices.at(i) - stock2_prices.at(i);
        spread[idx] = current_spread + spread[idx - 2];
        spread[idx + 1] = (current_spread * current_spread) + spread[idx - 1];
    }

    /*const size_t idx_n = (N - 1) * 2;
    double mean = spread[idx_n] / N;
    double stddev = std::sqrt(spread[idx_n + 1] / N - mean * mean);*/

    for (size_t i = N; i < stock1_prices.size(); ++i) {
        const size_t idx_curr = (i - 1) * 2;
        const size_t idx_prev = idx_curr - (N * 2);
        double mean = (spread[idx_curr] - spread[idx_prev]) / N;
        double stddev = std::sqrt((spread[idx_curr + 1] - spread[idx_prev + 1]) / N - mean * mean);
        double current_spread = stock1_prices.at(i) - stock2_prices.at(i);
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