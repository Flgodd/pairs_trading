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


template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, N> spread;
    size_t spread_index = 0;

    spread[0] = stock1_prices[0] - stock2_prices[0];
    spread[1] = stock1_prices[1] - stock2_prices[1];
    spread[2] = stock1_prices[2] - stock2_prices[2];
    spread[3] = stock1_prices[3] - stock2_prices[3];
    spread[4] = stock1_prices[4] - stock2_prices[4];
    spread[5] = stock1_prices[5] - stock2_prices[5];
    spread[6] = stock1_prices[6] - stock2_prices[6];
    spread[7] = stock1_prices[7] - stock2_prices[7];
    spread[8] = stock1_prices[8] - stock2_prices[8];
    spread[9] = stock1_prices[9] - stock2_prices[9];
    spread[10] = stock1_prices[10] - stock2_prices[10];
    spread[11] = stock1_prices[11] - stock2_prices[11];
    spread[12] = stock1_prices[12] - stock2_prices[12];
    spread[13] = stock1_prices[13] - stock2_prices[13];
    spread[14] = stock1_prices[14] - stock2_prices[14];
    spread[15] = stock1_prices[15] - stock2_prices[15];
    spread[16] = stock1_prices[16] - stock2_prices[16];
    spread[17] = stock1_prices[17] - stock2_prices[17];
    spread[18] = stock1_prices[18] - stock2_prices[18];
    spread[19] = stock1_prices[19] - stock2_prices[19];
    spread[20] = stock1_prices[20] - stock2_prices[20];
    spread[21] = stock1_prices[21] - stock2_prices[21];
    spread[22] = stock1_prices[22] - stock2_prices[22];
    spread[23] = stock1_prices[23] - stock2_prices[23];
    spread[24] = stock1_prices[24] - stock2_prices[14];
    spread[25] = stock1_prices[25] - stock2_prices[25];
    spread[26] = stock1_prices[26] - stock2_prices[26];
    spread[27] = stock1_prices[27] - stock2_prices[27];
    spread[28] = stock1_prices[28] - stock2_prices[28];

    //vector<int> check(4, 0);
    for(size_t i = N; i < stock1_prices.size(); ++i) {
        __m256d sum_vec = _mm256_setzero_pd();
        __m256d sq_sum_vec = _mm256_setzero_pd();


        __m256d spread_vec = _mm256_loadu_pd(&spread[0]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[4]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[8]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[12]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[16]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[20]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        spread_vec = _mm256_loadu_pd(&spread[24]);
        sum_vec = _mm256_add_pd(sum_vec, spread_vec);
        sq_sum_vec = _mm256_fmadd_pd(spread_vec, spread_vec, sq_sum_vec);
        //sq_sum_vec = _mm256_add_pd(sq_sum_vec, _mm256_mul_pd(spread_vec, spread_vec));

        __m256d temp1 = _mm256_hadd_pd(sum_vec, sum_vec);
        __m256d sum_vec_total = _mm256_add_pd(temp1, _mm256_permute2f128_pd(temp1, temp1, 0x1));

        __m256d temp2 = _mm256_hadd_pd(sq_sum_vec, sq_sum_vec);
        __m256d sq_sum_vec_total = _mm256_add_pd(temp2, _mm256_permute2f128_pd(temp2, temp2, 0x1));

        double final_sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sum_vec_total));
        double final_sq_sum = _mm_cvtsd_f64(_mm256_castpd256_pd128(sq_sum_vec_total));
        /*if(i == N+1){
            double t = 0;
            for(double s : spread){
                t += s;
            }
            cout<<t<<":"<<final_sum<<endl;
        }*/
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

        //if(i==8)cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<":"<<sum[0]<<endl;

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

BENCHMARK_TEMPLATE(BM_PairsTradingStrategyOptimized, 28);

BENCHMARK_MAIN();
