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

template<size_t I, size_t N, typename ArrayType>
inline void accumulateSpread(ArrayType& spread, const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    if constexpr (I < N) {
        double current_spread = stock1_prices[I] - stock2_prices[I];
        spread[I][0] = current_spread + spread[I - 1][0];
        spread[I][1] = (current_spread * current_spread) + spread[I - 1][1];
        accumulateSpread<I + 1, N>(spread, stock1_prices, stock2_prices);
    }
}


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<std::array<double, 2>, 1256> spread;
    vector<int> check(4, 0);

    spread[0][0] = stock1_prices[0] - stock2_prices[0];
    spread[0][1] = (stock1_prices[0] - stock2_prices[0])*(stock1_prices[0] - stock2_prices[0]);

    accumulateSpread<1, N>(spread, stock1_prices, stock2_prices);

    for(size_t i = N; i<1256; i++){
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double old_spread = stock1_prices[i-N] - stock2_prices[i-N];
        spread[i][0] = current_spread + spread[i-1][0] - (old_spread);
        spread[i][1] = (current_spread*current_spread) + spread[i-1][1] - (old_spread*old_spread);

    }

    for (size_t i = N; i < stock1_prices.size(); ++i) {

        double mean = spread[i-1][0]/ N;
        double stddev = std::sqrt(spread[i-1][1]/ N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;


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