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
    static_assert(N % 4 == 0, "N should be a multiple of 4 for AVX instructions");

    std::array<double, 1256> spread;
    const size_t size = stock1_prices.size();

    for (size_t i = 0; i < size; i += 4) {
        __m256d stock1_vec = _mm256_loadu_pd(&stock1_prices[i]);
        __m256d stock2_vec = _mm256_loadu_pd(&stock2_prices[i]);
        __m256d spread_vec = _mm256_sub_pd(stock1_vec, stock2_vec);
        _mm256_storeu_pd(&spread[i], spread_vec);
    }

    vector<int>check(4);

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d minus_one = _mm256_set1_pd(-1.0);
    const __m256d zero_point_eight = _mm256_set1_pd(0.8);
    const __m256d n_vec = _mm256_set1_pd(static_cast<double>(N));

    for (size_t i = N; i < size; ++i) {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d sq_sum_vec = _mm256_setzero_pd();

        for (size_t j = i - N; j < i; j += 4) {
            __m256d spread_vec = _mm256_loadu_pd(&spread[j]);
            sum_vec = _mm256_add_pd(sum_vec, spread_vec);
            sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        }

        __m256d temp1 = _mm256_hadd_pd(sum_vec, sum_vec);
        __m256d sum_vec_total = _mm256_add_pd(temp1, _mm256_permute2f128_pd(temp1, temp1, 0x1));

        __m256d temp2 = _mm256_hadd_pd(sq_sum_vec, sq_sum_vec);
        __m256d sq_sum_vec_total = _mm256_add_pd(temp2, _mm256_permute2f128_pd(temp2, temp2, 0x1));

        double sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sum_vec_total));
        double sq_sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sq_sum_vec_total));

        __m256d mean_vec = _mm256_div_pd(sum_vec_total, n_vec);
        __m256d stddev_vec = _mm256_sqrt_pd(_mm256_sub_pd(_mm256_div_pd(sq_sum_vec_total, n_vec), _mm256_mul_pd(mean_vec, mean_vec)));

        double current_spread = spread[i];
        __m256d current_spread_vec = _mm256_set1_pd(current_spread);
        __m256d z_score_vec = _mm256_div_pd(_mm256_sub_pd(current_spread_vec, mean_vec), stddev_vec);

        __m256d abs_z_score_vec = _mm256_max_pd(z_score_vec, _mm256_sub_pd(_mm256_setzero_pd(), z_score_vec));

        __m256d long_short_mask = _mm256_cmp_pd(z_score_vec, one, _CMP_GT_OQ);
        __m256d short_long_mask = _mm256_cmp_pd(z_score_vec, minus_one, _CMP_LT_OQ);
        __m256d close_positions_mask = _mm256_cmp_pd(abs_z_score_vec, zero_point_eight, _CMP_LT_OQ);

        int long_short_mask_bits = _mm256_movemask_pd(long_short_mask);
        int short_long_mask_bits = _mm256_movemask_pd(short_long_mask);
        int close_positions_mask_bits = _mm256_movemask_pd(close_positions_mask);
        // Perform trading actions based on the comparison results
        // ...
        if(long_short_mask_bits != 0);check[0]++;
        else if(short_long_mask_bits != 0);check[1]++;
        else if(close_positions_mask_bits != 0);check[2]++;
        else ;check[3]++;

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
