#include <benchmark/benchmark.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
//#include <immintrin.h>'
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
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, 1256> spread;
    //vector<int> check(4, 0);

    for(size_t i = 0; i < 1256; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    for (size_t i = N; i < stock1_prices.size(); ++i) {

        __m256d sum_vec = _mm256_setzero_pd();
        __m256d sq_sum_vec = _mm256_setzero_pd();

        for(size_t j = i-N; j < i; j += 4) {
            __m256d spread_vec = _mm256_loadu_pd(&spread[j]);
            sum_vec = _mm256_add_pd(sum_vec, spread_vec);
            sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
            //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));
        }

        __m256d temp1 = _mm256_hadd_pd(sum_vec, sum_vec);
        __m256d sum_vec_total = _mm256_add_pd(temp1, _mm256_permute2f128_pd(temp1, temp1, 0x1));

        __m256d temp2 = _mm256_hadd_pd(sq_sum_vec, sq_sum_vec);
        __m256d sq_sum_vec_total = _mm256_add_pd(temp2, _mm256_permute2f128_pd(temp2, temp2, 0x1));

        double sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sum_vec_total));
        double sq_sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sq_sum_vec_total));

        double mean = sum / N;
        double stddev = std::sqrt(sq_sum / N - mean * mean);
        double current_spread = spread[i];
        double z_score = (current_spread - mean) / stddev;


        if (z_score > 1.0) {
            //check[0]++;  // Long and Short
        } else if (z_score < -1.0) {
            //check[1]++;  // Short and Long
        } else if (std::abs(z_score) < 0.8) {
            //check[2]++;  // Close positions
        } else {
            //check[3]++;  // No signal
        }

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
