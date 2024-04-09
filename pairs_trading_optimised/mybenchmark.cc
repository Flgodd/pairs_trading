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

template<size_t N, size_t UnrollFactor>
struct LoopUnroll {
    static void computeSpread(std::array<double, N>& spread, const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices, size_t startIndex, double& sum, double& sq_sum) {
        __m256d spread_vec = _mm256_sub_pd(_mm256_loadu_pd(&stock1_prices[startIndex]), _mm256_loadu_pd(&stock2_prices[startIndex]));
        _mm256_storeu_pd(&spread[startIndex], spread_vec);

        __m256d sum_vec = _mm256_loadu_pd(&sum);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        _mm256_storeu_pd(&sum, sum_vec);

        __m256d sq_sum_vec = _mm256_loadu_pd(&sq_sum);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        _mm256_storeu_pd(&sq_sum, sq_sum_vec);

        LoopUnroll<N, UnrollFactor - 4>::computeSpread(spread, stock1_prices, stock2_prices, startIndex + 4, sum, sq_sum);
    }
};

template<size_t N>
struct LoopUnroll<N, 0> {
    static void computeSpread(std::array<double, N>& spread, const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices, size_t startIndex, double& sum, double& sq_sum) {
        // Base case, do nothing
    }
};

template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 4 == 0, "N should be a multiple of 4 for AVX instructions");
    std::array<double, N> spread;
    size_t spread_index = 0;
    double sum = 0.0;
    double sq_sum = 0.0;

    LoopUnroll<N, N>::computeSpread(spread, stock1_prices, stock2_prices, 0, sum, sq_sum);

    for (size_t i = N; i < stock1_prices.size(); ++i) {
        double mean = sum / N;
        double variance = sq_sum / N - mean * mean;
        double stddev = std::sqrt(variance);

        __m256d current_spread_vec = _mm256_sub_pd(_mm256_loadu_pd(&stock1_prices[i]), _mm256_loadu_pd(&stock2_prices[i]));
        double current_spread = _mm256_cvtsd_f64(current_spread_vec);

        __m256d z_score_vec = _mm256_div_pd(_mm256_sub_pd(current_spread_vec, _mm256_set1_pd(mean)), _mm256_set1_pd(stddev));
        double z_score = _mm256_cvtsd_f64(z_score_vec);

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

        __m256d old_value_vec = _mm256_set1_pd(old_value);
        __m256d sum_vec = _mm256_sub_pd(_mm256_set1_pd(sum), old_value_vec);
        sum_vec = _mm256_add_pd(sum_vec, current_spread_vec);
        sum = _mm256_cvtsd_f64(sum_vec);

        __m256d sq_sum_vec = _mm256_sub_pd(_mm256_set1_pd(sq_sum), _mm256_mul_pd(old_value_vec, old_value_vec));
        sq_sum_vec = _mm256_fmadd_pd(current_spread_vec, current_spread_vec, sq_sum_vec);
        sq_sum = _mm256_cvtsd_f64(sq_sum_vec);

        spread_index = (spread_index + 1) % N;
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
