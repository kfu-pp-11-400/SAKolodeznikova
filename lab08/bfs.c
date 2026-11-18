#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

typedef struct {
    int n;
    int m;
    int *offsets;
    int *edges;
} Graph;

/* построение CSR по списку рёбер */
static void build_csr(int n, int m, const int *eu, const int *ev, int directed, Graph *g) {
    g->n = n;
    g->m = directed ? m : 2 * m;
    g->offsets = calloc(n + 1, sizeof(int));
    g->edges = malloc((g->m > 0 ? g->m : 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        g->offsets[eu[i] + 1]++;
        if (!directed) g->offsets[ev[i] + 1]++;
    }
    for (int v = 0; v < n; v++) g->offsets[v + 1] += g->offsets[v];
    int *pos = malloc(n * sizeof(int));
    memcpy(pos, g->offsets, n * sizeof(int));
    for (int i = 0; i < m; i++) {
        g->edges[pos[eu[i]]++] = ev[i];
        if (!directed) g->edges[pos[ev[i]]++] = eu[i];
    }
    free(pos);
}

static int read_graph(const char *file, int directed, Graph *g) {
    FILE *f = fopen(file, "r");
    if (!f) return 0;
    int n, m;
    if (fscanf(f, "%d %d", &n, &m) != 2) { fclose(f); return 0; }
    int *eu = malloc(m * sizeof(int));
    int *ev = malloc(m * sizeof(int));
    for (int i = 0; i < m; i++) {
        if (fscanf(f, "%d %d", &eu[i], &ev[i]) != 2 ||
            eu[i] < 0 || eu[i] >= n || ev[i] < 0 || ev[i] >= n) {
            fclose(f); free(eu); free(ev);
            return 0;
        }
    }
    fclose(f);
    build_csr(n, m, eu, ev, directed, g);
    free(eu); free(ev);
    return 1;
}

static unsigned long long rng_state;

static double next_rand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}

/* случайный граф */
static int gen_graph(int n, int m, int seed, int skew, int directed, Graph *g) {
    if (n <= 0 || m < 0) return 0;
    rng_state = 88172645463325252ULL + (unsigned long long)seed * 2654435761ULL;
    for (int i = 0; i < 20; i++) next_rand();
    int *eu = malloc((m > 0 ? m : 1) * sizeof(int));
    int *ev = malloc((m > 0 ? m : 1) * sizeof(int));
    for (int i = 0; i < m; i++) {
        double r = next_rand();
        eu[i] = skew ? (int)(n * r * r * r) : (int)(n * r);
        ev[i] = (int)(n * next_rand());
    }
    build_csr(n, m, eu, ev, directed, g);
    free(eu); free(ev);
    return 1;
}

static void bfs_seq(const Graph *g, int s, int *dist) {
    int *queue = malloc(g->n * sizeof(int));
    int head = 0, tail = 0;
    for (int v = 0; v < g->n; v++) dist[v] = -1;
    dist[s] = 0;
    queue[tail++] = s;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
            int u = g->edges[j];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    free(queue);
}

static void push(int **list, int *size, int *cap, int u) {
    if (*size == *cap) {
        *cap *= 2;
        *list = realloc(*list, *cap * sizeof(int));
    }
    (*list)[(*size)++] = u;
}

/* гибридный BFS */
static int bfs_hybrid(const Graph *g, int s, int *dist, int rank, int nprocs, int *front_sizes, long *my_count) {
    int n = g->n;
    int nthreads = omp_get_max_threads();
    int *frontier = malloc(n * sizeof(int));
    int *next_frontier = malloc(n * sizeof(int));
    int *local_candidate = malloc(n * sizeof(int));
    int *global_candidate = malloc(n * sizeof(int));
    int **thread_candidates = malloc(nthreads * sizeof(int *));
    int *tsize = calloc(nthreads, sizeof(int));
    int *tcap = malloc(nthreads * sizeof(int));
    for (int t = 0; t < nthreads; t++) {
        tcap[t] = 16;
        thread_candidates[t] = malloc(tcap[t] * sizeof(int));
    }

    for (int v = 0; v < n; v++) dist[v] = -1;
    dist[s] = 0;
    frontier[0] = s;
    int frontier_size = 1;
    int level = 0;

    while (frontier_size > 0) {
        front_sizes[level] = frontier_size;
        int local_frontier_size = 0;
        if (frontier_size > rank)
            local_frontier_size = (frontier_size - rank + nprocs - 1) / nprocs;
        *my_count += local_frontier_size;

        for (int t = 0; t < nthreads; t++) tsize[t] = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < local_frontier_size; k++) {
                int idx = rank + k * nprocs;
                int v = frontier[idx];
                for (int j = g->offsets[v]; j < g->offsets[v + 1]; j++) {
                    int u = g->edges[j];
                    if (dist[u] == -1)
                        push(&thread_candidates[tid], &tsize[tid], &tcap[tid], u);
                }
            }
        }

        memset(local_candidate, 0, n * sizeof(int));
        for (int t = 0; t < nthreads; t++)
            for (int i = 0; i < tsize[t]; i++)
                local_candidate[thread_candidates[t][i]] = 1;

        MPI_Allreduce(local_candidate, global_candidate, n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        int next_size = 0;
        for (int u = 0; u < n; u++) {
            if (global_candidate[u] == 1 && dist[u] == -1) {
                dist[u] = level + 1;
                next_frontier[next_size++] = u;
            }
        }

        int *tmp = frontier; frontier = next_frontier; next_frontier = tmp;
        frontier_size = next_size;
        level++;
    }

    for (int t = 0; t < nthreads; t++) free(thread_candidates[t]);
    free(thread_candidates); free(tsize); free(tcap);
    free(frontier); free(next_frontier); free(local_candidate); free(global_candidate);
    return level;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) printf("usage: %s graph.txt|rand:N:M:seed[:skew] start [directed] [static] [stats]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int directed = 0, use_static = 0, stats = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "directed") == 0) directed = 1;
        else if (strcmp(argv[i], "static") == 0) use_static = 1;
        else if (strcmp(argv[i], "stats") == 0) stats = 1;
    }
    int s = atoi(argv[2]);

    Graph g;
    int ok;
    int gn, gm, gseed, gskew = 0;
    if (strncmp(argv[1], "rand:", 5) == 0)
        ok = sscanf(argv[1] + 5, "%d:%d:%d:%d", &gn, &gm, &gseed, &gskew) >= 3 && gen_graph(gn, gm, gseed, gskew, directed, &g);
    else
        ok = read_graph(argv[1], directed, &g);
    if (!ok || s < 0 || s >= g.n) {
        if (rank == 0) printf("cannot read/generate graph or bad start vertex\n");
        MPI_Finalize();
        return 1;
    }
    int n = g.n;
    int threads = omp_get_max_threads();
    if (use_static) omp_set_schedule(omp_sched_static, 0);
    else omp_set_schedule(omp_sched_dynamic, 1);

    int *dist_seq = malloc(n * sizeof(int));
    int *dist_par = malloc(n * sizeof(int));
    int *front_sizes = malloc((n + 1) * sizeof(int));
    long my_count = 0;
    double t_seq = 0.0;

    if (rank == 0) {
        bfs_seq(&g, s, dist_seq);
        t_seq = MPI_Wtime();
        bfs_seq(&g, s, dist_seq);
        t_seq = MPI_Wtime() - t_seq;
    }

    bfs_hybrid(&g, s, dist_par, rank, nprocs, front_sizes, &my_count);
    my_count = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double t_local = MPI_Wtime();
    int levels = bfs_hybrid(&g, s, dist_par, rank, nprocs, front_sizes, &my_count);
    t_local = MPI_Wtime() - t_local;
    double t_par = 0.0;
    MPI_Reduce(&t_local, &t_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int *dist_root = malloc(n * sizeof(int));
    if (rank == 0) memcpy(dist_root, dist_par, n * sizeof(int));
    MPI_Bcast(dist_root, n, MPI_INT, 0, MPI_COMM_WORLD);
    int diff_local = 0, diff_all = 0;
    for (int v = 0; v < n; v++)
        if (dist_root[v] != dist_par[v]) diff_local++;
    MPI_Allreduce(&diff_local, &diff_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    long *counts = malloc(nprocs * sizeof(long));
    MPI_Gather(&my_count, 1, MPI_LONG, counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int unreachable = 0, mismatch = 0, max_front = 0;
        for (int v = 0; v < n; v++) {
            if (dist_par[v] == -1) unreachable++;
            if (dist_seq[v] != dist_par[v]) mismatch++;
        }
        for (int l = 0; l < levels; l++)
            if (front_sizes[l] > max_front) max_front = front_sizes[l];

        printf("graph: V=%d E=%d (csr %d) start=%d %s\n", n, directed ? g.m : g.m / 2, g.m, s, directed ? "directed" : "undirected");
        printf("levels=%d max_front=%d unreachable=%d mismatches=%d same_on_all_procs=%s\n", levels, max_front, unreachable, mismatch, diff_all == 0 ? "YES" : "NO");
        printf("processes=%d threads=%d total=%d schedule=%s\n", nprocs, threads, nprocs * threads, use_static ? "static" : "dynamic,1");
        printf("processed vertices per process:");
        for (int r = 0; r < nprocs; r++) printf(" %ld", counts[r]);
        printf("\n");
        if (stats) {
            printf("front sizes:");
            for (int l = 0; l < levels; l++) printf(" %d", front_sizes[l]);
            printf("\n");
        }
        double speedup = t_seq / t_par;
        printf("T_seq=%.6f T_par=%.6f speedup=%.3f efficiency=%.3f\n", t_seq, t_par, speedup, speedup / (nprocs * threads));
        printf("correct=%s\n", (mismatch == 0 && diff_all == 0) ? "YES" : "NO");
    }

    free(dist_seq); free(dist_par); free(dist_root); free(front_sizes); free(counts);
    free(g.offsets); free(g.edges);
    MPI_Finalize();
    return 0;
}
