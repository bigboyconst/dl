#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#define BUILD_DIR "build/"
#define SRC_DIR "src/"
#define THIRDPARTY_DIR "src/thirdparty/"
#define CULA_SRC SRC_DIR"cuLA/cuLA.cu"
#define CULA_INCL SRC_DIR"cuLA/"

#include <stdbool.h>

// This is so fucking cursed bro
typedef struct {
	const char*** items;
	size_t count;
	size_t capacity;
} Sources;

Sources sources = {0};

const char* targets[] = {
	"cudatest",
	"nn"
};

size_t source_sizes[] = {
	1, 
	2
};

void nvcc(Cmd* cmd, const char* target, const char** sources, size_t count) {
	cmd_append(cmd, "nvcc", "-o", temp_sprintf(BUILD_DIR"%s", target));
	cmd_append(cmd, "-I"THIRDPARTY_DIR);
	cmd_append(cmd, "-lcublas");

	for (size_t i = 0; i < count; i++) {
		cmd_append(cmd, temp_sprintf(SRC_DIR"%s/%s", target, sources[i]));
	}
}

void nvcc_cuLA(Cmd* cmd, const char* target, const char** sources, size_t count) {
	cmd_append(cmd, "nvcc", "-o", temp_sprintf(BUILD_DIR"%s", target));
	cmd_append(cmd, "-I"THIRDPARTY_DIR);
	cmd_append(cmd, "-I"CULA_INCL);
	cmd_append(cmd, "-lcublas");
	for (size_t i = 0; i < count; i++) {
		cmd_append(cmd, temp_sprintf(SRC_DIR"%s/%s", target, sources[i]));
	}
	cmd_append(cmd, CULA_SRC);
}

int main(int argc, char** argv) {
	NOB_GO_REBUILD_URSELF(argc, argv);
	// cudatest
	const char* cudatest_srcs[] = { "cudatest.cu", "../cuLA/cuLA.cu" };
	da_append(&sources, cudatest_srcs);

	// nn
	const char* nn_srcs[] = { "nn.cpp", "NeuralNetwork.cpp" };
	da_append(&sources, nn_srcs);

	Cmd cmd = {0};
	
	String_Builder sb = {0};

	if (!mkdir_if_not_exists(BUILD_DIR)) return 1;

	// nvcc(&cmd, targets[0], sources.items[0]);

	// Temporary fix cause im too lazy to track down the issue

	nvcc_cuLA(&cmd, targets[0], sources.items[0], source_sizes[0]);

	if (!cmd_run_sync_and_reset(&cmd)) return 1;

	nvcc_cuLA(&cmd, targets[1], sources.items[1], source_sizes[1]);

	if (!cmd_run_sync_and_reset(&cmd)) return 1;

	return 0; 
}