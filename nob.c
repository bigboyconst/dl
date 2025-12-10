#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#include "nob.h"

#define BUILD_DIR "build/"
#define SRC_DIR "src/"
#define THIRDPARTY_DIR "src/thirdparty/"

const char* targets[] = {
	"cudatest"
};

Cmd cmd = {0};

// TODO: Extend to take in multiple source files (a single one isn't really my style)
void nvcc(const char* target, const char* extension) {
	const char* bin_name = temp_sprintf(BUILD_DIR"%s", target);
	const char* src_name = temp_sprintf(SRC_DIR"%s/%s%s", target, target, extension);
	cmd_append(&cmd, "nvcc", 
		"-o", bin_name,
		"-I"THIRDPARTY_DIR,
		src_name
	);
}

int main(int argc, char** argv) {
	NOB_GO_REBUILD_URSELF(argc, argv);

	if (!mkdir_if_not_exists(BUILD_DIR)) return 1;

	nvcc(targets[0], ".cu");

	if (!cmd_run_sync_and_reset(&cmd)) return 1;

	return 0; 
}