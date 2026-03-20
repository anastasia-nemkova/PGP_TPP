#include <iostream>
#include <iomanip>
#include <cmath>

void solve(float a, float b, float c) {
    float D = b * b - 4 * a * c;

    if (D < 0) {
        std::cout << "imaginary" << "\n";
        return;
    } else if (D == 0) {
        float x = -b / (2 * a);
        std::cout << std::fixed << std::setprecision(6) << x << "\n";
        return;
    } else {
        float x1 = (-b + sqrt(D)) / (2 * a);
        float x2 = (-b - sqrt(D)) / (2 * a);
        std::cout << std::fixed << std::setprecision(6) << x1 << " " << x2 << "\n";
        return;
    }
}

int main() {
    
    std::ios::sync_with_stdio(false);
    std::cin.tie(0); std::cout.tie(0);

    float a, b, c;

    std::cin >> a >> b >> c;

    if (a == 0 && b == 0 && c == 0) {
        std::cout << "any" << "\n";
    } else if (a == 0 && b == 0) {
        std::cout << "incorrect" << "\n";
    } else if (a == 0) {
        std::cout << - c / b << "\n"; 
    } else {
        solve(a, b, c);
    }

}