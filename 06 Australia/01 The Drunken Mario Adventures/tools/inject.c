/*
 * inject.c - run a Python file inside the Marathon Lab process.
 *
 * Marathon Lab is a Nuitka build: the game modules (lab.level, lab.fitness,
 * lab.engine, lab.brain, lab.evolution) are compiled to C and live inside
 * marathon-lab.bin together with an embedded CPython 3.12. The executable
 * exports the Python C-API (its numpy/pygame extension modules need it), so
 * a preloaded library can wait for the interpreter to come up and then run
 * ordinary Python code in it - with the real, compiled game modules
 * importable as usual.
 *
 * Build:  gcc -O2 -shared -fPIC -o inject.so inject.c -ldl -lpthread
 * Use:    INJECT_PY=probe.py INJECT_DELAY_MS=2500 LD_PRELOAD=./inject.so \
 *         SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy ./marathon-lab.bin
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef int (*is_init_fn)(void);
typedef int (*gil_ensure_fn)(void);
typedef void (*gil_release_fn)(int);
typedef int (*run_string_fn)(const char *);

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    size_t got = buf ? fread(buf, 1, (size_t)n, f) : 0;
    fclose(f);
    if (!buf) return NULL;
    buf[got] = 0;
    return buf;
}

static void *worker(void *arg) {
    (void)arg;
    const char *path = getenv("INJECT_PY");
    const char *delay = getenv("INJECT_DELAY_MS");
    int delay_ms = delay ? atoi(delay) : 2500;

    is_init_fn is_init = NULL;
    for (int i = 0; i < 1200 && !is_init; i++) {
        is_init = (is_init_fn)dlsym(RTLD_DEFAULT, "Py_IsInitialized");
        if (!is_init) usleep(50000);
    }
    if (!is_init) {
        fprintf(stderr, "[inject] Py_IsInitialized not found\n");
        return NULL;
    }
    while (!is_init()) usleep(20000);
    usleep((useconds_t)delay_ms * 1000);  /* let main.py import the lab package */

    gil_ensure_fn gil_ensure = (gil_ensure_fn)dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    gil_release_fn gil_release = (gil_release_fn)dlsym(RTLD_DEFAULT, "PyGILState_Release");
    run_string_fn run_string = (run_string_fn)dlsym(RTLD_DEFAULT, "PyRun_SimpleString");
    char *code = read_file(path);
    if (!code || !gil_ensure || !gil_release || !run_string) {
        fprintf(stderr, "[inject] setup failed (code=%p)\n", (void *)code);
        return NULL;
    }
    int state = gil_ensure();
    int rc = run_string(code);
    gil_release(state);
    free(code);
    fprintf(stderr, "[inject] %s finished, rc=%d\n", path, rc);
    return NULL;
}

__attribute__((constructor)) static void start_injector(void) {
    if (!getenv("INJECT_PY")) return;
    pthread_t thread;
    if (pthread_create(&thread, NULL, worker, NULL) == 0) pthread_detach(thread);
}
