#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef float T;

void transpose_rows(const T *A, T *B, size_t N) {
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

void transpose_cols(const T *A, T *B, size_t N) {
    for (size_t j = 0; j < N; j++)
        for (size_t i = 0; i < N; i++)
            B[j * N + i] = A[i * N + j];
}

void transpose_blocked(const T *A, T *B, size_t N, size_t b) {
    for (size_t ii = 0; ii < N; ii += b) {
        size_t imax = ii + b < N ? ii + b : N;
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jmax = jj + b < N ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = jj; j < jmax; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_rows_omp(const T *A, T *B, size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

void transpose_rows_omp_chunk(const T *A, T *B, size_t N, int chunk) {
    #pragma omp parallel for schedule(static, chunk)
    for (long long i = 0; i < (long long)N; i++)
        for (size_t j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

void transpose_blocked_omp(const T *A, T *B, size_t N, size_t b) {
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < (long long)N; ii += b) {
        size_t imax = ii + b < N ? ii + b : N;
        for (size_t jj = 0; jj < N; jj += b) {
            size_t jmax = jj + b < N ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = jj; j < jmax; j++)
                    B[j * N + i] = A[i * N + j];
        }
    }
}

void transpose_inplace(T *A, size_t N, size_t b) {
    for (size_t ii = 0; ii < N; ii += b) {
        size_t imax = ii + b < N ? ii + b : N;
        for (size_t jj = ii; jj < N; jj += b) {
            size_t jmax = jj + b < N ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = (jj == ii ? i + 1 : jj); j < jmax; j++) {
                    T tmp = A[i * N + j];
                    A[i * N + j] = A[j * N + i];
                    A[j * N + i] = tmp;
                }
        }
    }
}

void transpose_inplace_omp(T *A, size_t N, size_t b) {
    #pragma omp parallel for schedule(dynamic)
    for (long long ii = 0; ii < (long long)N; ii += b) {
        size_t imax = ii + b < N ? ii + b : N;
        for (size_t jj = ii; jj < N; jj += b) {
            size_t jmax = jj + b < N ? jj + b : N;
            for (size_t i = ii; i < imax; i++)
                for (size_t j = (jj == (size_t)ii ? i + 1 : jj); j < jmax; j++) {
                    T tmp = A[i * N + j];
                    A[i * N + j] = A[j * N + i];
                    A[j * N + i] = tmp;
                }
        }
    }
}

void run(const char *v, const T *A, T *B, size_t N, size_t b) {
    if (strcmp(v, "rows") == 0) transpose_rows(A, B, N);
    else if (strcmp(v, "cols") == 0) transpose_cols(A, B, N);
    else if (strcmp(v, "blocked") == 0) transpose_blocked(A, B, N, b);
    else if (strcmp(v, "rows_omp") == 0) transpose_rows_omp(A, B, N);
    else if (strcmp(v, "rows_omp_static1") == 0) transpose_rows_omp_chunk(A, B, N, 1);
    else if (strcmp(v, "rows_omp_static8") == 0) transpose_rows_omp_chunk(A, B, N, 8);
    else if (strcmp(v, "blocked_omp") == 0) transpose_blocked_omp(A, B, N, b);
    else if (strcmp(v, "inplace") == 0) transpose_inplace(B, N, b);
    else if (strcmp(v, "inplace_omp") == 0) transpose_inplace_omp(B, N, b);
}

int main(int argc, char *argv[]) {
    size_t N = atoll(argv[1]);
    size_t b = atoll(argv[2]);
    int threads = atoi(argv[3]);
    const char *version = argv[4];
    omp_set_num_threads(threads);

    T *A = malloc(N * N * sizeof(T));
    T *B = malloc(N * N * sizeof(T));
    T *Bref = malloc(N * N * sizeof(T));
    for (size_t k = 0; k < N * N; k++) A[k] = (T)k;

    transpose_rows(A, Bref, N);

    int inplace = strncmp(version, "inplace", 7) == 0;

    if (inplace) memcpy(B, A, N * N * sizeof(T));
    run(version, A, B, N, b);

    int reps = inplace ? 1 : 1 + (int)(200000000ULL / (N * N));

    double times[5];
    for (int r = 0; r < 5; r++) {
        if (inplace) memcpy(B, A, N * N * sizeof(T));
        double t = omp_get_wtime();
        for (int k = 0; k < reps; k++)
            run(version, A, B, N, b);
        times[r] = (omp_get_wtime() - t) / reps;
    }
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
    double med = times[2];

    size_t bad = 0;
    for (size_t k = 0; k < N * N; k++)
        if (B[k] != Bref[k]) bad++;

    double bw = 2.0 * N * N * sizeof(T) / med / 1e9;
    int used = strstr(version, "omp") ? threads : 1;

    printf("N=%zu b=%zu threads=%d version=%s time=%.4f s BW=%.2f GB/s check=%s\n",
           N, b, used, version, med, bw, bad == 0 ? "OK" : "FAIL");

    free(A);
    free(B);
    free(Bref);
    return 0;
}
