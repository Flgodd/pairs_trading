#include <array>
#include <chrono>
#include <cstdlib>
#include <experimental/simd>
#include <iostream>
#include <random>

using std::experimental::fixed_size_simd;
using Vec3D_v = fixed_size_simd<float, 3>;

using Vec3D = std::array<float, 3>;
float scalar_product (const std::array<float, 3> &a, const std::array<float, 3> &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

int main () {
    constexpr std::size_t VECREG_SIZE = fixed_size_simd<float, 3>::size ();
    std::array<Vec3D, VECREG_SIZE * 1000> arr;
    std::array<Vec3D_v, VECREG_SIZE * 1000> arr_v;
    std::random_device rd;
    std::mt19937 generator (rd ());
    std::uniform_real_distribution<float> distribution (0.f, 1.f);

    for (std::size_t i = 0; i < arr.size (); ++i) {
        arr[i] = {distribution (generator), distribution (generator), distribution (generator) };

        for (int j = 0; j < 3; ++j)
            arr_v[i][j] = arr[i][j];
    }

    Vec3D_v result_v;

    for (int iter = 0; iter < 3; ++iter) {

        for (int j = 0; j < 3; ++j)
            result_v[j] = 0.f;

        auto start = std::chrono::high_resolution_clock::now ();

        for (std::size_t i = 1; i < arr.size (); ++i) {
            result_v += arr_v[i - 1] * arr_v[i];
        }

        float result = std::experimental::reduce (result_v);
        auto end = std::chrono::high_resolution_clock::now ();
        auto elapsed = end - start;
        std::cout << "VC: " << elapsed.count () << " (result: " << result << ")" << std::endl;

        result = 0;
        start = std::chrono::high_resolution_clock::now ();

        for (std::size_t i = 1; i < arr.size (); ++i) {
            result += scalar_product (arr[i - 1], arr[i]);
        }

        end = std::chrono::high_resolution_clock::now ();
        elapsed = end - start;
        std::cout << "notVC: " << elapsed.count () << " (result: " << result << ")" << std::endl;
    }

    return EXIT_SUCCESS;
}