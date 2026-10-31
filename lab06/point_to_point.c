#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TAG_MESSAGE 1
#define TAG_REPLY 2
#define MSG_LEN 64

static void basic(int rank) {
    char msg[MSG_LEN];
    if (rank == 0) {
        strcpy(msg, "Hello from process 0");
        printf("[rank 0] send to 1: \"%s\" (tag=%d)\n", msg, TAG_MESSAGE);
        MPI_Send(msg, (int)strlen(msg) + 1, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank 0] received from 1: \"%s\" (tag=%d)\n", msg, TAG_REPLY);
    } else if (rank == 1) {
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank 1] received from 0: \"%s\" (tag=%d)\n", msg, TAG_MESSAGE);
        strcpy(msg, "Message received");
        printf("[rank 1] send to 0: \"%s\" (tag=%d)\n", msg, TAG_REPLY);
        MPI_Send(msg, (int)strlen(msg) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

static void all(int rank, int size) {
    char msg[MSG_LEN];
    if (rank == 0) {
        int sent = 0, received = 0, ok = 1;
        for (int r = 1; r < size; r++) {
            snprintf(msg, MSG_LEN, "Hello to process %d", r);
            printf("[rank 0] send to %d: \"%s\" (tag=%d)\n", r, msg, TAG_MESSAGE);
            MPI_Send(msg, (int)strlen(msg) + 1, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
            sent++;
        }
        for (int r = 1; r < size; r++) {
            MPI_Status st;
            MPI_Recv(msg, MSG_LEN, MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, &st);
            printf("[rank 0] received from %d: \"%s\" (tag=%d)\n", st.MPI_SOURCE, msg, st.MPI_TAG);
            received++;
            if (st.MPI_SOURCE != r) ok = 0;
        }
        printf("P=%d sent=%d received=%d all_replies=%s\n", size, sent, received,
               (sent == size - 1 && received == size - 1 && ok) ? "YES" : "NO");
    } else {
        MPI_Recv(msg, MSG_LEN, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[rank %d] received from 0: \"%s\" (tag=%d)\n", rank, msg, TAG_MESSAGE);
        snprintf(msg, MSG_LEN, "Message received by %d", rank);
        MPI_Send(msg, (int)strlen(msg) + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

static void pingpong(int rank, int n, int reps) {
    int warmup = 10;
    char *buf = malloc(n);
    memset(buf, 'a', n);
    double t = 0.0;
    for (int k = 0; k < warmup + reps; k++) {
        if (k == warmup) t = MPI_Wtime();
        if (rank == 0) {
            MPI_Send(buf, n, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
            MPI_Recv(buf, n, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(buf, n, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(buf, n, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
        }
    }
    if (rank == 0) {
        t = MPI_Wtime() - t;
        printf("size=%d reps=%d time=%.9f\n", n, reps, t / reps);
    }
    free(buf);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }
    if (argc < 2) {
        if (rank == 0) printf("usage: %s basic | all | size n reps\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (strcmp(argv[1], "basic") == 0) {
        basic(rank);
    } else if (strcmp(argv[1], "all") == 0) {
        all(rank, size);
    } else if (strcmp(argv[1], "size") == 0 && argc >= 4) {
        pingpong(rank, atoi(argv[2]), atoi(argv[3]));
    } else {
        if (rank == 0) printf("unknown mode\n");
    }

    MPI_Finalize();
    return 0;
}