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

Cmd cmd = {0};

void init_sources() {
	// cudatest
	const char* cudatest_srcs[] = { "cudatest.cu" };
	da_append(&sources, cudatest_srcs);
}

void nvcc(const char* target, const char** sources) {
	const char* bin_name = temp_sprintf(BUILD_DIR"%s", target);
	cmd_append(&cmd, "nvcc", 
		"-o", bin_name,
		"-I"THIRDPARTY_DIR
	);
	for (size_t i = 0; i < NOB_ARRAY_LEN(sources); i++) {
		const char* src_name = temp_sprintf(SRC_DIR"%s/%s", target, sources[i]);
		cmd_append(&cmd, src_name);
	}
}

int main(int argc, char** argv) {
	NOB_GO_REBUILD_URSELF(argc, argv);

	init_sources();

	if (!mkdir_if_not_exists(BUILD_DIR)) return 1;

	nvcc(targets[0], sources.items[0]);

	if (!cmd_run_sync_and_reset(&cmd)) return 1;

	return 0; 
}