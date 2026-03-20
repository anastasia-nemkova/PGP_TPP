#include <iostream>
#include <iomanip>

void bubble_sort(float *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(0); std::cout.tie(0);

    int n;
    std::cin >> n;

    float* arr = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }

    bubble_sort(arr, n);

    std::cout << std::setprecision(6) << std::scientific;
    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }

    std::cout << "\n";

    free(arr);
    return 0;
}