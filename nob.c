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
	cmd_append(&cmd, "nvcc", 
		"-o", target, 
		"-I"THIRDPARTY_DIR,
		temp_sprintf(SRC_DIR"%s/%s.%s", target, target, extension)
	);
}

int main(int argc, char** argv) {
	NOB_GO_REBUILD_URSELF(argc, argv);

	if (!mkdir_if_not_exists(BUILD_DIR)) return 1;

	nvcc(targets[0], ".cu");
	
}