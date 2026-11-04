/*
 * Формат аргументов: ./matmul N b threads version
 *   N - размер матрицы
 *   b - размер блока 
 *   threads - число потоков OpenMP
 *   version - ijk_seq | ijk_omp | ikj_seq | ikj_omp | blocked_omp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void ijk_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

void ijk_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

void ikj_seq(const double *A, const double *B, double *C, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void ikj_omp(const double *A, const double *B, double *C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

void blocked_omp(const double *A, const double *B, double *C, int N, int b) {
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += b)
        for (int jj = 0; jj < N; jj += b) {
            int imax = ii + b < N ? ii + b : N;
            int jmax = jj + b < N ? jj + b : N;
            for (int kk = 0; kk < N; kk += b) {
                int kmax = kk + b < N ? kk + b : N;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        double a = A[i * N + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
            }
        }
}

void run(const char *v, const double *A, const double *B, double *C, int N, int b) {
    if (strcmp(v, "ijk_seq") == 0) ijk_seq(A, B, C, N);
    else if (strcmp(v, "ijk_omp") == 0) ijk_omp(A, B, C, N);
    else if (strcmp(v, "ikj_seq") == 0) ikj_seq(A, B, C, N);
    else if (strcmp(v, "ikj_omp") == 0) ikj_omp(A, B, C, N);
    else if (strcmp(v, "blocked_omp") == 0) blocked_omp(A, B, C, N, b);
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    int b = atoi(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];
    omp_set_num_threads(threads);

    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = malloc(bytes);
    double *B = malloc(bytes);
    double *C = malloc(bytes);
    double *Cref = malloc(bytes);

    for (size_t k = 0; k < (size_t)N * N; k++) {
        A[k] = (k % 10) / 10.0;
        B[k] = (k % 7) / 7.0;
    }

    ijk_seq(A, B, Cref, N);

    int reps = 1 + (int)(200000000LL / ((long long)N * N * N));

    double times[5];
    for (int r = 0; r < 5; r++) {
        double total = 0.0;
        for (int k = 0; k < reps; k++) {
            memset(C, 0, bytes);
            double t = omp_get_wtime();
            run(version, A, B, C, N, b);
            total += omp_get_wtime() - t;
        }
        times[r] = total / reps;
    }
    
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
    double med = times[2];

    double maxdiff = 0.0;
    for (size_t k = 0; k < (size_t)N * N; k++) {
        double d = fabs(C[k] - Cref[k]);
        if (d > maxdiff) maxdiff = d;
    }

    int used = strstr(version, "omp") ? threads : 1;
    printf("N=%d b=%d threads=%d version=%s time=%.4f check=%s\n",
           N, b, used, version, med, maxdiff < 1e-9 ? "OK" : "FAIL");

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}