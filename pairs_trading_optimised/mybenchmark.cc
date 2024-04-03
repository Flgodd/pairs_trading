//RUN ON P3 Isambard
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
//#include "experimental/simd"
#include <experimental/simd>
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
//#include <arm_neon.h>
#include <array>



using namespace std;

namespace stdx = std::experimental;

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
    static_assert(N % 2 == 0, "N should be a multiple of 2 for SIMD operations");

    std::array<double, N> spread;
    size_t spread_index = 0;

    for(size_t i = 0; i < N; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    std::vector<int> check(4, 0);
    for(size_t i = N; i < stock1_prices.size(); ++i) {

        using Vd = stdx::simd<double>; // Use the default SIMD ABI for doubles
        Vd sum_vec(0.0), sq_sum_vec(0.0);

        for(size_t j = 0; j < N; j += 4) {
            Vd spread_vec(&spread[j], stdx::element_aligned_tag{}); // Load spread values into a SIMD vector
            sum_vec += spread_vec;
            sq_sum_vec += spread_vec * spread_vec;
        }

        double final_sum = stdx::reduce(sum_vec);
        double final_sq_sum = stdx::reduce(sq_sum_vec);

        double mean = final_sum / N;
        double stddev = std::sqrt(final_sq_sum / N - mean * mean);

        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;

        spread[spread_index] = current_spread;

        if(z_score > 1.0) {
            // Long and Short
            //check[0]++;
        } else if(z_score < -1.0) {
            // Short and Long
            //check[1]++;
        } else if (std::abs(z_score) < 0.8) {
            // Close positions
            //check[2]++;
        } else {
            // No signal
            //check[3]++;
        }

        spread_index = (spread_index + 1) % N;
    }
    //cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<endl;

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





