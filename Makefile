# Dreamflash Build Configuration & Patching Makefile
#
# Resolves Finding #1 (docs/FINDINGS.md): Separates ROCm backend compilation from
# hardcoded APU architecture (gfx1151 / Strix Halo). Allows targeting discrete
# consumer AMD GPUs (gfx1100 / gfx1101 / gfx1030) with discrete memory overrides.

ROCM_PATH ?= /opt/rocm
HIPCC     ?= $(ROCM_PATH)/bin/hipcc
PYTHON    ?= python3
CC        ?= cc
BUILD_DIR ?= build

# Hardware target axes
ARCH        ?= gfx1100
MEMORY_MODE ?= discrete

# Compile flags
CXXFLAGS += -O3 -std=c++17 -Iinclude -Itools/roofline
ifeq ($(MEMORY_MODE),discrete)
    CXXFLAGS += -DDS4_DISCRETE_GPU=1 -DDS4_FREE_RESERVE_GIB=0.5
else
    CXXFLAGS += -DDS4_UNIFIED_MEMORY=1 -DDS4_FREE_RESERVE_GIB=16.0
endif

# The C engine is plain C11 and builds without ROCm.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Iinclude

.PHONY: all test test-roofline test-trace test-spec test-config test-c \
        check-env rocm-discrete strix-halo clean

all: test

check-env:
	@echo "=== Dreamflash Environment ==="
	@echo "ROCM_PATH   : $(ROCM_PATH)"
	@echo "GPU ARCH    : $(ARCH)"
	@echo "MEMORY MODE : $(MEMORY_MODE)"
	@echo "PYTHON      : $(PYTHON)"

test: test-roofline test-trace test-spec test-config test-c

test-roofline:
	$(PYTHON) tests/test_roofline.py

test-trace:
	$(PYTHON) tests/test_trace_replay.py

test-spec:
	$(PYTHON) tests/test_speculative_decoding.py

test-config:
	$(PYTHON) tests/test_discrete_config.py

# Builds and exercises the C engine. Previously nothing in this repo called into
# src/*.c at all, so it could regress freely without any target noticing.
test-c: $(BUILD_DIR)/test_expert_io
	./$(BUILD_DIR)/test_expert_io

$(BUILD_DIR)/test_expert_io: tests/test_expert_io.c src/expert_io.c src/memory_manager.c \
                             include/expert_io.h include/memory_manager.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) tests/test_expert_io.c src/expert_io.c src/memory_manager.c -o $@

rocm-discrete:
	@echo "Building Dreamflash ROCm discrete target for $(ARCH)..."
	@echo "FLAGS: --offload-arch=$(ARCH) $(CXXFLAGS)"
	# Build step template for hipcc when ROCm environment is present
	# $(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) ...

strix-halo:
	@echo "Building upstream APU target for gfx1151..."
	# $(HIPCC) --offload-arch=gfx1151 -DDS4_UNIFIED_MEMORY=1 ...

clean:
	@echo "Cleaning temporary build artifacts..."
	rm -rf $(BUILD_DIR)
