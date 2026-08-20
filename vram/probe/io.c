/* Storage probe: what a residency engine can actually pull off this disk.
 *
 *   cc -O2 -pthread -o io vram/probe/io.c && ./io /path/on/ext4/scratch
 *
 * Two questions. Sequential O_DIRECT is the number people quote. The one
 * that decides a demand-paging design is 64 KiB random reads -- lmz's block
 * size, and the granularity an expert or an adapter is paged at -- swept
 * over queue depth, because at depth one the disk is a latency device and
 * at depth 64 it is a bandwidth device, and on this box those differ by 19x.
 *
 * Writes the scratch file itself and unlinks it. Needs ~4 GiB free.
 * O_DIRECT bypasses the page cache, so no root and no cache dropping.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#define TOTAL (4ull << 30)
#define SEQ_BS (4ull << 20)
#define RND_BS (64ull << 10)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int FD;
static size_t NBLK, ITER;

static void *reader(void *arg) {
    size_t id = (size_t)arg;
    void *b;
    if (posix_memalign(&b, 4096, RND_BS)) return NULL;
    /* xorshift, seeded per thread: threads must not read the same blocks or
     * the second one measures the first one's cache. */
    unsigned long s = 88172645463325252ull + id * 2654435761ull;
    for (size_t i = 0; i < ITER; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        if (pread(FD, b, RND_BS, (off_t)(s % NBLK) * RND_BS) != (ssize_t)RND_BS) {
            perror("pread");
            break;
        }
    }
    free(b);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: io <scratch-path>\n"); return 2; }
    const char *path = argv[1];

    void *buf;
    if (posix_memalign(&buf, 4096, SEQ_BS)) return 1;
    memset(buf, 7, SEQ_BS);

    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }
    for (size_t o = 0; o < TOTAL; o += SEQ_BS)
        if (write(fd, buf, SEQ_BS) != (ssize_t)SEQ_BS) { perror("write"); return 1; }
    fsync(fd);
    close(fd);

    FD = open(path, O_RDONLY | O_DIRECT);
    if (FD < 0) { perror("open O_DIRECT"); return 1; }
    NBLK = TOTAL / RND_BS;

    double t = now();
    for (size_t o = 0; o < TOTAL; o += SEQ_BS)
        if (pread(FD, buf, SEQ_BS, o) != (ssize_t)SEQ_BS) { perror("pread"); return 1; }
    printf("O_DIRECT seq 4 MiB, 1 thread : %6.2f GB/s\n", TOTAL / (now() - t) / 1e9);

    int depths[] = {1, 2, 4, 8, 16, 32, 64};
    pthread_t th[64];
    for (unsigned k = 0; k < sizeof(depths) / sizeof(*depths); k++) {
        int T = depths[k];
        ITER = 40000 / T;
        if (ITER < 200) ITER = 200;
        t = now();
        for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, reader, (void *)(size_t)i);
        for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
        double dt = now() - t, bytes = (double)T * ITER * RND_BS;
        printf("O_DIRECT rnd 64 KiB, QD=%-3d  : %6.2f GB/s  %8.0f IOPS  %6.1f us\n",
               T, bytes / dt / 1e9, T * ITER / dt, dt / (T * ITER) * 1e6 * T);
    }
    close(FD);
    unlink(path);
    free(buf);
    return 0;
}
