#include <arm_neon.h>

void pairs_trading_strategy_optimized(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    static_assert(N % 2 == 0, "N should be a multiple of 2 for NEON instructions");

    std::array<double, N> spread;
    size_t spread_index = 0;

    // Calculate spread for the first N elements
    for (size_t i = 0; i < N; ++i) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    // Calculate sum, squared sum, and mean using NEON instructions
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    float64x2_t sq_sum_vec = vdupq_n_f64(0.0);
    for (size_t i = 0; i < N; i += 2) {
        float64x2_t spread_vec = vld1q_f64(&spread[i]);
        sum_vec = vaddq_f64(sum_vec, spread_vec);
        sq_sum_vec = vaddq_f64(sq_sum_vec, vmulq_f64(spread_vec, spread_vec));
    }

    double sum[2], sq_sum[2];
    vst1q_f64(sum, sum_vec);
    vst1q_f64(sq_sum, sq_sum_vec);
    double final_sum = sum[0] + sum[1];
    double final_sq_sum = sq_sum[0] + sq_sum[1];

    double mean = final_sum / N;
    double inv_sqrt_n = 1.0 / sqrt(N);

    // Loop through remaining elements of stock1_prices
    for (size_t i = N; i < stock1_prices.size(); ++i) {
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) * inv_sqrt_n;

        // Update spread and check array efficiently
        spread[spread_index] = current_spread;
        if (z_score > 1.0) {
            check[0]++;
        } else if (z_score < -1.0) {
            check[1]++;
        } else if (std::abs(z_score) < 0.8) {
            check[2]++;
        } else {
            check[3]++;
        }

        spread_index = (spread_index + 1) % N;
    }

    std::cout << check[0] << ":" << check[1] << ":" << check[2] << ":" << check[3] << std::endl;
}

