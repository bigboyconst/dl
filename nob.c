#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#define BUILD_DIR "build/"
#define SRC_DIR "src/"
#define THIRDPARTY_DIR "src/thirdparty/"

// This is so fucking cursed bro
typedef struct {
	const char*** items;
	size_t count;
	size_t capacity;
} Sources;

Sources sources = {0};

const char* targets[] = {
	"cudatest"
};

void nvcc(Cmd* cmd, const char* target, const char** sources) {
	cmd_append(cmd, "nvcc", "-o", temp_sprintf(BUILD_DIR"%s", target));
	cmd_append(cmd, "-I"THIRDPARTY_DIR);
	cmd_append(cmd, "-lcublas");

	for (size_t i = 0; i < NOB_ARRAY_LEN(sources); i++) {
		cmd_append(cmd, temp_sprintf(SRC_DIR"%s/%s", target, sources[i]));
	}
}

int main(int argc, char** argv) {
	NOB_GO_REBUILD_URSELF(argc, argv);
	// cudatest
	const char* cudatest_srcs[] = { "cudatest.cu", "../cuLA/cuLA.cu" };
	da_append(&sources, cudatest_srcs);

	Cmd cmd = {0};
	
	String_Builder sb = {0};

	if (!mkdir_if_not_exists(BUILD_DIR)) return 1;

	nvcc(&cmd, targets[0], sources.items[0]);

	if (!cmd_run_sync_and_reset(&cmd)) return 1;

	return 0; 
}