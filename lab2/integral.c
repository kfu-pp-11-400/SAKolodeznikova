#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <windows.h>

// Функция G: f(x) = sqrt(2 + sin x), [0; pi/2]
double f(double x) {
    return sqrt(2.0 + sin(x));
}

double integral_seq(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }
    return h * sum;
}

double integral_par(double a, double b, long long n) {
    double h = (b - a) / n;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }
    return h * sum;
}

double find_eps(double a, double b, double eps, int parallel, long long *n_res) {
    long long n = 1000;
    double prev, cur;

    prev = parallel ? integral_par(a, b, n) : integral_seq(a, b, n);
    while (1) {
        n *= 2;
        cur = parallel ? integral_par(a, b, n) : integral_seq(a, b, n);
        if (fabs(cur - prev) < eps) break;
        prev = cur;
    }
    *n_res = n;
    return cur;
}

int main(int argc, char *argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    double a = atof(argv[1]);
    double b = atof(argv[2]);
    long long n = atoll(argv[3]);
    int threads = atoi(argv[4]);
    omp_set_num_threads(threads);

    int used = 0;
    #pragma omp parallel
    {
        #pragma omp single
        used = omp_get_num_threads();
    }

    printf("a = %.15f, b = %.15f, n = %lld\n\n", a, b, n);

    double t = omp_get_wtime();
    double i1 = integral_seq(a, b, n);
    double t_seq = omp_get_wtime() - t;

    printf("Последовательная версия:\n");
    printf("  интеграл = %.15f\n", i1);
    printf("  время    = %.4f с\n", t_seq);
    printf("  потоков  = 1\n\n");

    t = omp_get_wtime();
    double i2 = integral_par(a, b, n);
    double t_par = omp_get_wtime() - t;

    printf("Параллельная версия (OpenMP):\n");
    printf("  интеграл = %.15f\n", i2);
    printf("  время    = %.4f с\n", t_par);
    printf("  потоков  = %d\n\n", used);

    printf("Ускорение = %.2f\n", t_seq / t_par);

    if (argc > 5) {
        double eps = atof(argv[5]);
        long long n_seq, n_par;

        printf("\nТочность eps = %g\n\n", eps);

        t = omp_get_wtime();
        double e1 = find_eps(a, b, eps, 0, &n_seq);
        double te_seq = omp_get_wtime() - t;

        printf("Последовательная версия:\n");
        printf("  интеграл = %.15f\n", e1);
        printf("  n        = %lld\n", n_seq);
        printf("  время    = %.4f с\n\n", te_seq);

        t = omp_get_wtime();
        double e2 = find_eps(a, b, eps, 1, &n_par);
        double te_par = omp_get_wtime() - t;

        printf("Параллельная версия (OpenMP):\n");
        printf("  интеграл = %.15f\n", e2);
        printf("  n        = %lld\n", n_par);
        printf("  время    = %.4f с\n", te_par);
        printf("  потоков  = %d\n", used);
    }
    return 0;
}