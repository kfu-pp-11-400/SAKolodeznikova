#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static inline double soft(double z, double a) {
    if (z > a) return z - a;
    if (z < -a) return z + a;
    return 0.0;
}

// цикл с ветвлениями в soft() не векторизуется
static void update(double *w, double *b, const double *g, double gb, int N, int d, double eta, double lambda) {
    for (int j = 0; j < d; j++)
        w[j] = soft(w[j] - eta * g[j] / N, eta * lambda);
    *b -= eta * gb / N;
}

// векторизация только автоматическая
void train_seq(const double *X, const double *y, double *w, double *b, int N, int d, int K, double eta, double lambda) {
    double *g = malloc(d * sizeof(double));
    for (int it = 0; it < K; it++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            for (int j = 0; j < d; j++)
                pred += x[j] * w[j];
            double e = pred - y[i];
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
            gb += e;
        }
        update(w, b, g, gb, N, d, eta, lambda);
    }
    free(g);
}

// внутренние циклы по признакам векторизуются директивами omp simd
void train_simd(const double *X, const double *y, double *w, double *b, int N, int d, int K, double eta, double lambda) {
    double *g = malloc(d * sizeof(double));
    for (int it = 0; it < K; it++) {
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            const double *x = X + (size_t)i * d;
            double pred = *b;
            #pragma omp simd reduction(+:pred)
            for (int j = 0; j < d; j++)
                pred += x[j] * w[j];
            double e = pred - y[i];
            #pragma omp simd
            for (int j = 0; j < d; j++)
                g[j] += e * x[j];
            gb += e;
        }
        update(w, b, g, gb, N, d, eta, lambda);
    }
    free(g);
}

// параллелизм по объектам (цикл по i). Каждый поток копит частичный градиент в своём буфере, затем буферы объединяются последовательно
void train_omp(const double *X, const double *y, double *w, double *b, int N, int d, int K, double eta, double lambda) {
    int T = omp_get_max_threads();
    int ld = d + 8;
    double *part = malloc((size_t)T * ld * sizeof(double));
    double *g = malloc(d * sizeof(double));
    for (int it = 0; it < K; it++) {
        #pragma omp parallel
        {
            double *gl = part + (size_t)omp_get_thread_num() * ld;
            memset(gl, 0, ld * sizeof(double));
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const double *x = X + (size_t)i * d;
                double pred = *b;
                for (int j = 0; j < d; j++)
                    pred += x[j] * w[j];
                double e = pred - y[i];
                for (int j = 0; j < d; j++)
                    gl[j] += e * x[j];
                gl[d] += e;
            }
        }

        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int t = 0; t < T; t++) {
            const double *gl = part + (size_t)t * ld;
            for (int j = 0; j < d; j++)
                g[j] += gl[j];
            gb += gl[d];
        }
        update(w, b, g, gb, N, d, eta, lambda);
    }
    free(part);
    free(g);
}

void train_omp_simd(const double *X, const double *y, double *w, double *b, int N, int d, int K, double eta, double lambda) {
    int T = omp_get_max_threads();
    int ld = d + 8;
    double *part = malloc((size_t)T * ld * sizeof(double));
    double *g = malloc(d * sizeof(double));
    for (int it = 0; it < K; it++) {
        #pragma omp parallel
        {
            double *gl = part + (size_t)omp_get_thread_num() * ld;
            memset(gl, 0, ld * sizeof(double));
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                const double *x = X + (size_t)i * d;
                double pred = *b;
                #pragma omp simd reduction(+:pred)
                for (int j = 0; j < d; j++)
                    pred += x[j] * w[j];
                double e = pred - y[i];
                // Векторизация накопления градиента в приватный буфер gl
                #pragma omp simd
                for (int j = 0; j < d; j++)
                    gl[j] += e * x[j];
                gl[d] += e;
            }
        }
        memset(g, 0, d * sizeof(double));
        double gb = 0.0;
        for (int t = 0; t < T; t++) {
            const double *gl = part + (size_t)t * ld;
            // Векторизация суммирования буферов потоков по признакам j
            #pragma omp simd
            for (int j = 0; j < d; j++)
                g[j] += gl[j];
            gb += gl[d];
        }
        update(w, b, g, gb, N, d, eta, lambda);
    }
    free(part);
    free(g);
}

void run(const char *v, const double *X, const double *y, double *w, double *b, int N, int d, int K, double eta, double lambda) {
    if (strcmp(v, "seq") == 0) train_seq(X, y, w, b, N, d, K, eta, lambda);
    else if (strcmp(v, "omp") == 0) train_omp(X, y, w, b, N, d, K, eta, lambda);
    else if (strcmp(v, "simd") == 0) train_simd(X, y, w, b, N, d, K, eta, lambda);
    else if (strcmp(v, "omp_simd") == 0) train_omp_simd(X, y, w, b, N, d, K, eta, lambda);
}

double loss(const double *X, const double *y, const double *w, double b, int N, int d, double lambda) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = b;
        for (int j = 0; j < d; j++)
            pred += X[(size_t)i * d + j] * w[j];
        s += (pred - y[i]) * (pred - y[i]);
    }
    double l1 = 0.0;
    for (int j = 0; j < d; j++)
        l1 += fabs(w[j]);
    return s / (2.0 * N) + lambda * l1;
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        printf("usage: %s N d K lambda threads version [eta]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int d = atoi(argv[2]);
    int K = atoi(argv[3]);
    double lambda = atof(argv[4]);
    int threads = atoi(argv[5]);
    const char *version = argv[6];
    double eta = argc > 7 ? atof(argv[7]) : 0.5;
    omp_set_num_threads(threads);
 
    double *X = malloc((size_t)N * d * sizeof(double));
    double *y = malloc((size_t)N * sizeof(double));
    double *wtrue = malloc(d * sizeof(double));
    double *w = malloc(d * sizeof(double));
    double *wref = malloc(d * sizeof(double));
    double b, bref;
 
    srand(12345);
    for (int j = 0; j < d; j++)
        wtrue[j] = (j % 4 == 0) ? ((j % 8 == 0) ? 2.0 : -1.5) : 0.0;
    for (int i = 0; i < N; i++) {
        double s = 0.5;
        for (int j = 0; j < d; j++) {
            double v = 2.0 * rand() / RAND_MAX - 1.0;
            X[(size_t)i * d + j] = v;
            s += v * wtrue[j];
        }
        y[i] = s + 0.1 * (2.0 * rand() / RAND_MAX - 1.0);
    }
 
    memset(wref, 0, d * sizeof(double));
    bref = 0.0;
    train_seq(X, y, wref, &bref, N, d, K, eta, lambda);
    double Fref = loss(X, y, wref, bref, N, d, lambda);

    int reps = 1 + (int)(500000000LL / ((long long)N * d * K));
    double times[5];
    for (int r = 0; r < 5; r++) {
        double total = 0.0;
        for (int k = 0; k < reps; k++) {
            memset(w, 0, d * sizeof(double));
            b = 0.0;
            double t = omp_get_wtime();
            run(version, X, y, w, &b, N, d, K, eta, lambda);
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
 
    double dw = 0.0;
    for (int j = 0; j < d; j++) {
        double diff = fabs(w[j] - wref[j]);
        if (diff > dw) dw = diff;
    }
    double F = loss(X, y, w, b, N, d, lambda);
 
    int zeros = 0;
    for (int j = 0; j < d; j++)
        if (fabs(w[j]) < 1e-6) zeros++;
 
    int used = strstr(version, "omp") ? threads : 1;
    printf("N=%d d=%d K=%d lambda=%g eta=%g threads=%d version=%s loss=%.10f dw=%.3e zeros=%d time=%.4f check=%s\n",
           N, d, K, lambda, eta, used, version, F, dw, zeros, med, (dw < 1e-8 && fabs(F - Fref) < 1e-10) ? "OK" : "FAIL");
 
    free(X);
    free(y);
    free(wtrue);
    free(w);
    free(wref);
    return 0;
}