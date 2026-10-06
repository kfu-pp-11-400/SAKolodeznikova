#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double pi_seq(long long n) {
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

double pi_par(long long n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        sum += sign / (2.0 * i + 1.0);
    }
    return 4.0 * sum;
}

int main(int argc, char *argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    long long n = atoll(argv[1]);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    int used = 0;
    #pragma omp parallel
    {
        #pragma omp single
        used = omp_get_num_threads();
    }

    printf("n = %lld\n\n", n);

    double t = omp_get_wtime();
    double p1 = pi_seq(n);
    double t_seq = omp_get_wtime() - t;

    printf("Последовательная версия:\n");
    printf("  pi      = %.15f\n", p1);
    printf("  ошибка  = %.3e\n", fabs(p1 - M_PI));
    printf("  время   = %.4f с\n", t_seq);
    printf("  потоков = 1\n\n");

    t = omp_get_wtime();
    double p2 = pi_par(n);
    double t_par = omp_get_wtime() - t;

    printf("Параллельная версия (OpenMP):\n");
    printf("  pi      = %.15f\n", p2);
    printf("  ошибка  = %.3e\n", fabs(p2 - M_PI));
    printf("  время   = %.4f с\n", t_par);
    printf("  потоков = %d\n\n", used);

    printf("Ускорение = %.2f\n", t_seq / t_par);
    return 0;
}