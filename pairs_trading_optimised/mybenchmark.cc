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
//#include <experimental/execution_policy>
#include <chrono>
//#include <experimental/numeric>
#include <arm_neon.h>
#include <array>
#include <thread>
#include <omp.h>


#define NUM_THREADS 8


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

void parallelUpSweep(vector<double>& x) {
    int n = x.size();
    int maxDepth = std::log2(n);

    for (int d = 0; d < maxDepth; ++d) {
        int stride = std::pow(2, d);
#pragma omp parallel for
        for (int k = stride - 1; k < n; k += 2 * stride) {
            x[k + stride] += x[k];
        }
    }
}

void parallelDownSweep(vector<double>& x) {
    int n = x.size();
    x[n - 1] = 0;
    int maxDepth = std::log2(n);

    for (int d = maxDepth - 1; d >= 0; --d) {
        int stride = std::pow(2, d);
#pragma omp parallel for
        for (int k = stride - 1; k < n; k += 2 * stride) {
            double tmp = x[k];
            x[k] = x[k + stride];
            x[k + stride] += tmp;
        }
    }
}

void recursive_blelloch(vector<double>& x, int depth) {
    int n = x.size();
    int numThreads = NUM_THREADS;

    if (depth == 0 || n <= 2 * numThreads) {
        parallelUpSweep(x);
        parallelDownSweep(x);
        return;
    }

    int div = n / (2 * numThreads);
    int rem = n % (2 * numThreads);
    if (rem != 0) {
        x.resize(n + (2 * numThreads - rem), 0);
        div++;
    }

    vector<vector<double>> toHoldValues(div);
    vector<double> newX(div);

#pragma omp parallel for
    for (int i = 0; i < div; i++) {
        int start = 2 * numThreads * i;
        int end = std::min(start + 2 * numThreads, static_cast<int>(x.size()));
        vector<double> temp(x.begin() + start, x.begin() + end);
        parallelUpSweep(temp);
        parallelDownSweep(temp);
        toHoldValues[i] = temp;
        newX[i] = temp.back() + x[end - 1];
    }

    double bigg = newX.back();
    recursive_blelloch(newX, depth - 1);

    x.clear();
    newX.push_back(newX.back() + bigg);

#pragma omp parallel for
    for (int i = 0; i < div; i++) {
        double nx = newX[i];
        for (int j = 0; j < toHoldValues[i].size(); j++) {
            toHoldValues[i][j] += nx;
        }
    }

    x.reserve(n);
    for (int i = 0; i < div; i++) {
        x.insert(x.end(), toHoldValues[i].begin(), toHoldValues[i].end());
    }
}



template<size_t N>
void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    vector<double> spread_sum(1256);
    vector<double> spread_sq_sum(1256);
    vector<int> check(4, 0);


    for(int i = 0; i<stock1_prices.size(); i++){
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        spread_sum[i] = current_spread;
        spread_sq_sum[i] = current_spread*current_spread;
    }

    int depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    float  check_depth = std::log(spread_sum.size())/log(NUM_THREADS*2);
    int rem = (spread_sum.size()%(NUM_THREADS*2));

    if(rem != 0 || check_depth > depth)depth++;

    int xdepth = std::log(x.size())/log(NUM_THREADS*2);
    float  xcheck_depth = std::log(x.size())/log(NUM_THREADS*2);
    int xrem = (x.size()%(NUM_THREADS*2));

    if(xrem != 0 || xcheck_depth > xdepth)xdepth++;
    vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8
            1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8,1, 2, 3, 4, 5, 6, 7, 8};
    recursive_blelloch(x, xdepth);
    cout<<x[0]<<":"<<x[1]<<":"<<x[2]<<":"<<x[3]<<":"<<x[4]<<":"<<x[5]<<":"<<x[6]<<":"<<x[7]<<":"<<
            x[8]<<":"<<x[9]<<":"<<x[10]<<":"<<x[11]<<":"<<x[12]<<":"<<x[13]<<":"<<x[14]
            <<x[15]<<":"<<x[16]<<":"<<x[17]<<":"<<x[18]<<":"<<x[19]<<":"<<x[20]<<":"<<x[127]<<endl;

    recursive_blelloch(spread_sum, depth);
    recursive_blelloch(spread_sq_sum, depth);

    for (size_t i = N; i < stock1_prices.size(); ++i) {

        const double mean = (spread_sum[i] - spread_sum[i-N])/ N;
        const double stddev = std::sqrt((spread_sq_sum[i] - spread_sq_sum[i-N])/ N - mean * mean);
        const double current_spread = stock1_prices[i] - stock2_prices[i];
        const double z_score = (current_spread - mean) / stddev;


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

    //std::cout << "Maximum number of threads = " << omp_get_max_threads() << std::endl;

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





