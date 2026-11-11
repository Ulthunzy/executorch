# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ifneq ($(MAKECMDGOALS),clean)
ifndef QNN_INCLUDE
QNN_INCLUDE := $(QNN_SDK_ROOT)/include/QNN
endif
ifndef QNN_TARGET_LIB
QNN_TARGET_LIB := $(QNN_SDK_ROOT)/lib/aarch64-android
endif
ifndef HEXAGON_SDK_ROOT
$(error "ERROR: HEXAGON_SDK_ROOT is not set")
endif
ifndef ANDROID_NDK_ROOT
$(info "WARNING: ANDROID_NDK_ROOT is not set; htp_aarch64 will fail")
endif

ifeq ($(wildcard $(QNN_INCLUDE)),)
$(error "ERROR: QNN_INCLUDE path is invalid: $(QNN_INCLUDE)")
endif
ifeq ($(wildcard $(HEXAGON_SDK_ROOT)),)
$(error "ERROR: HEXAGON_SDK_ROOT path is invalid: $(HEXAGON_SDK_ROOT)")
endif
endif

HTP_BUILD_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
CUSTOM_OP_ROOT := $(abspath $(HTP_BUILD_ROOT)/..)
DELTANET_ROOT := $(CUSTOM_OP_ROOT)/deltanet_op_package_htp/DeltaNetOpPackage
ATTENTION_ROOT := $(CUSTOM_OP_ROOT)/attention_op_package_htp/AttentionOpPackage
RULE_FILES := $(abspath $(MAKEFILE_LIST))

# Keep the serialized package identity for existing PTE files.
PACKAGE_NAME := DeltaNetOpPackage
LIBRARY_NAME := libQnn$(PACKAGE_NAME).so
HTP_LIBRARY_NAME := libQnn$(PACKAGE_NAME)_HTP.so
WORK := build
SRC_DIR := $(HTP_BUILD_ROOT)/src
OP_SRC_DIR := $(DELTANET_ROOT)/src/ops
ATTENTION_SRC_DIR := $(ATTENTION_ROOT)/src/ops

OP_SOURCES := $(wildcard $(OP_SRC_DIR)/*.cpp) $(wildcard $(ATTENTION_SRC_DIR)/*.cpp)
OTHER_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OP_OBJS := $(patsubst $(OP_SRC_DIR)/%.cpp,ops/%.o,$(wildcard $(OP_SRC_DIR)/*.cpp))
OP_OBJS += $(patsubst $(ATTENTION_SRC_DIR)/%.cpp,attention/%.o,$(wildcard $(ATTENTION_SRC_DIR)/*.cpp))
OTHER_OBJS := $(patsubst $(SRC_DIR)/%,%,$(patsubst %.cpp,%.o,$(OTHER_SOURCES)))

HFILES := $(wildcard $(QNN_INCLUDE)/*.h) $(CUSTOM_OP_ROOT)/htp_arch.h $(wildcard $(OP_SRC_DIR)/*.h)
HFILES += $(wildcard $(DELTANET_ROOT)/src/kernels/*.h) $(wildcard $(DELTANET_ROOT)/src/utils/*.h)
HFILES += $(wildcard $(ATTENTION_SRC_DIR)/attention/*.h) $(wildcard $(HTP_BUILD_ROOT)/include/*.h)
HFILES += $(wildcard $(CUSTOM_OP_ROOT)/attention_support/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/*.h)
HFILES += $(wildcard $(QNN_INCLUDE)/HTP/core/*.h)

COMMON_CXX_FLAGS := -std=c++17 -I$(QNN_INCLUDE) -I$(HTP_BUILD_ROOT)/include -fPIC -Wall -Wreorder
COMMON_CXX_FLAGS += -Wno-missing-braces -Wno-unused-function -Werror
COMMON_CXX_FLAGS += -Wno-format -Wno-unused-command-line-argument
COMMON_CXX_FLAGS += -fvisibility=default -stdlib=libc++
COMMON_CXX_FLAGS += -DQNN_API="__attribute__((visibility(\"default\")))"
COMMON_CXX_FLAGS += -D__QAIC_HEADER_EXPORT="__attribute__((visibility(\"default\")))"

ifeq ($(wildcard $(HEXAGON_TOOLS_ROOT)/bin/hexagon-clang++),)
HEXAGON_TOOLS_DIR := $(HEXAGON_TOOLS_ROOT)/Tools
else
HEXAGON_TOOLS_DIR := $(HEXAGON_TOOLS_ROOT)
endif
ifeq ($(HEXAGON_TOOLS_ROOT),)
HEXAGON_TOOLS_DIR := $(HEXAGON_SDK_ROOT)/tools/HEXAGON_Tools/19.0.04/Tools
endif

X86_LIBNATIVE_RELEASE_DIR := $(HEXAGON_TOOLS_DIR)
X86_CXX ?= clang++-9
ifeq ($(shell $(X86_CXX) -v 2>&1 | grep -c "clang version"), 0)
  X86_CXX := clang++
endif
X86_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include
X86_C_FLAGS += -DUSE_OS_LINUX
X86_CXX_FLAGS := $(X86_CPPFLAGS) $(COMMON_CXX_FLAGS) $(X86_C_FLAGS) -fomit-frame-pointer
# The host container provides libstdc++ headers but not libc++ headers.
# Keep libc++ for Android/Hexagon targets and use the available host STL here.
X86_CXX_FLAGS += -stdlib=libstdc++
X86_CXX_FLAGS += -Wno-unused-const-variable
X86_CXX_FLAGS += -Wno-invalid-offsetof
X86_LDFLAGS := $(LDFLAGS) -stdlib=libstdc++ -Wl,--whole-archive
X86_LDFLAGS += -L$(X86_LIBNATIVE_RELEASE_DIR)/libnative/lib -lnative
X86_LDFLAGS += -Wl,--no-whole-archive -lpthread

HEXAGON_CXX := $(HEXAGON_TOOLS_DIR)/bin/hexagon-clang++
HEXAGON_CC := $(HEXAGON_TOOLS_DIR)/bin/hexagon-clang
HMX_ROOT := $(HEXAGON_SDK_ROOT)/libs/qhl_hmx
HMX_OBJECTS := hmx/qhblas_hmx_transfer_row_ahf.o hmx/qhblas_hmx_transfer_column_ahf.o hmx/qhblas_hmx_matrix_matrix_mpy_hf_sm.o
HEXAGON_BASE_FLAGS := $(COMMON_CXX_FLAGS) -mhvx -mhvx-length=128B -mhmx
HEXAGON_BASE_FLAGS += -DUSE_OS_QURT -O3 -Wno-reorder -DPREPARE_DISABLED
HEXAGON_BASE_FLAGS += $(HEXAGON_CPPFLAGS)
HEXAGON_LDFLAGS := -nostdlib++ -Wl,-Bstatic -lc++ -lc++abi -Wl,-Bdynamic

AARCH64_CXX := $(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
ARM_CLANG_OPTS := --target=aarch64-none-linux-android21
ARM_CLANG_OPTS += --sysroot=$(ANDROID_NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/sysroot
ARM_CLANG_OPTS += -stdlib=libc++ -static-libstdc++
AARCH64_C_FLAGS := -D__HVXDBL__ -I$(X86_LIBNATIVE_RELEASE_DIR)/libnative/include
AARCH64_C_FLAGS += -DUSE_OS_LINUX -DANDROID
AARCH64_CXX_FLAGS := $(COMMON_CXX_FLAGS) $(AARCH64_C_FLAGS) -fomit-frame-pointer
AARCH64_CXX_FLAGS += -Wno-invalid-offsetof -Wno-unused-variable
AARCH64_CXX_FLAGS += -Wno-unused-parameter -Wno-missing-braces -Wno-sign-compare
AARCH64_CXX_FLAGS += -Wno-unused-private-field -Wno-ignored-qualifiers
AARCH64_CXX_FLAGS += -Wno-missing-field-initializers
AARCH64_LDFLAGS := -L$(QNN_TARGET_LIB) -lQnnHtp -lQnnHtpPrepare

define COMMON_BUILD_CONFIG :=
package=$(PACKAGE_NAME)
sources=$(OP_SOURCES) $(OTHER_SOURCES)
qnn_include=$(realpath $(QNN_INCLUDE))
qnn_target_lib=$(realpath $(QNN_TARGET_LIB))
hexagon_sdk=$(realpath $(HEXAGON_SDK_ROOT))
hexagon_tools=$(realpath $(HEXAGON_TOOLS_DIR))
android_ndk=$(realpath $(ANDROID_NDK_ROOT))
endef

define X86_BUILD_CONFIG :=
$(COMMON_BUILD_CONFIG)
compiler=$(X86_CXX)
version=$(shell $(X86_CXX) --version 2>/dev/null)
flags=$(X86_CXX_FLAGS)
link_flags=$(X86_LDFLAGS)
endef

define AARCH64_BUILD_CONFIG :=
$(COMMON_BUILD_CONFIG)
compiler=$(AARCH64_CXX)
version=$(shell $(AARCH64_CXX) --version 2>/dev/null)
flags=$(ARM_CLANG_OPTS) $(AARCH64_CXX_FLAGS)
link_flags=$(AARCH64_LDFLAGS)
endef

# Architecture makefiles can append flags after including this file.
define HEXAGON_BUILD_CONFIG
$(COMMON_BUILD_CONFIG)
compiler=$(HEXAGON_CXX)
version=$(shell $(HEXAGON_CXX) --version 2>/dev/null)
flags=$(HEXAGON_BASE_FLAGS)
link_flags=$(HEXAGON_LDFLAGS)
endef

export X86_BUILD_CONFIG AARCH64_BUILD_CONFIG HEXAGON_BUILD_CONFIG

all: htp_x86 htp_aarch64 htp_v79

htp_x86: $(WORK)/x86_64-linux-clang/$(LIBRARY_NAME)
htp_aarch64: $(WORK)/aarch64-android/$(LIBRARY_NAME)
htp_v79: $(WORK)/hexagon-v79/$(HTP_LIBRARY_NAME)
htp_v81: $(WORK)/hexagon-v81/$(HTP_LIBRARY_NAME)
htp_v85: $(WORK)/hexagon-v85/$(HTP_LIBRARY_NAME)

test-attention: $(WORK)/x86_64-linux-clang/test_attention
	$<

test-attention-u16: $(WORK)/x86_64-linux-clang/test_attention_u16
	$<

test-attention-stream: $(WORK)/x86_64-linux-clang/test_attention_stream
	$<

test-chunk-state-update: $(WORK)/x86_64-linux-clang/test_chunk_state_update
	$<

test-hmx-tile-layout: $(WORK)/x86_64-linux-clang/test_hmx_tile_layout
	$<

test-chunk: $(WORK)/x86_64-linux-clang/test_chunk
	$<

test-op-package-interface: $(WORK)/x86_64-linux-clang/test_op_package_interface
	$<

$(WORK)/x86_64-linux-clang/test_op_package_interface: $(HTP_BUILD_ROOT)/tests/test_op_package_interface.cpp $(WORK)/x86_64-linux-clang/OpPackageInterface.o $(RULE_FILES)
	$(X86_CXX) $(X86_CXX_FLAGS) -O2 -UNDEBUG $< $(WORK)/x86_64-linux-clang/OpPackageInterface.o -o $@

$(WORK)/x86_64-linux-clang/test_chunk: $(DELTANET_ROOT)/tests/test_chunk.cpp $(DELTANET_ROOT)/src/kernels/Chunk.h $(DELTANET_ROOT)/src/utils/HmxTileLayout.h $(HTP_BUILD_ROOT)/include/HvxUtils.h $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -O2 -Werror -UNDEBUG $< -o $@ $(X86_LDFLAGS)

$(WORK)/x86_64-linux-clang/test_hmx_tile_layout: $(DELTANET_ROOT)/tests/test_hmx_tile_layout.cpp $(DELTANET_ROOT)/src/utils/HmxTileLayout.h $(HTP_BUILD_ROOT)/include/HvxUtils.h $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -O2 -Werror -UNDEBUG $< -o $@ $(X86_LDFLAGS)

$(WORK)/x86_64-linux-clang/test_chunk_state_update: $(DELTANET_ROOT)/tests/test_chunk_state_update.cpp $(HTP_BUILD_ROOT)/include/HvxUtils.h $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -O2 -Werror -UNDEBUG $< -o $@ $(X86_LDFLAGS)

$(WORK)/x86_64-linux-clang/test_attention_stream: $(ATTENTION_ROOT)/tests/test_attention_stream.cpp $(wildcard $(ATTENTION_SRC_DIR)/attention/*.h) $(HTP_BUILD_ROOT)/include/HmxDispatch.h $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -UNDEBUG -pthread $< -o $@

$(WORK)/x86_64-linux-clang/test_attention_u16: $(ATTENTION_ROOT)/tests/test_attention_u16.cpp $(wildcard $(ATTENTION_SRC_DIR)/attention/*.h) $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -O2 -UNDEBUG -pthread $< -o $@ $(X86_LDFLAGS)

$(WORK)/x86_64-linux-clang/test_attention: $(ATTENTION_ROOT)/tests/test_attention.cpp $(wildcard $(ATTENTION_SRC_DIR)/attention/*.h) $(HTP_BUILD_ROOT)/include/HmxDispatch.h $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) -std=c++17 -O2 -Wall -Wextra -Werror -UNDEBUG -pthread $< -o $@

test-reduction: $(WORK)/x86_64-linux-clang/test_reduction
	$<

test-reference-reduction: $(WORK)/x86_64-linux-clang/test_reference_reduction
	$<

$(WORK)/x86_64-linux-clang/test_reference_reduction: $(DELTANET_ROOT)/tests/test_reference_reduction.cpp $(wildcard $(DELTANET_ROOT)/src/kernels/*.h) $(wildcard $(DELTANET_ROOT)/src/utils/*.h) $(wildcard $(HTP_BUILD_ROOT)/include/*.h) $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -O2 -ffunction-sections -fdata-sections $< -Wl,--gc-sections $(X86_LDFLAGS) -o $@

test-fp16-register: $(WORK)/x86_64-linux-clang/test_fp16_reference $(WORK)/x86_64-linux-clang/test_fp16_register
	$(WORK)/x86_64-linux-clang/test_fp16_reference $(WORK)/x86_64-linux-clang/fp16-reference.bin
	$(WORK)/x86_64-linux-clang/test_fp16_register $(WORK)/x86_64-linux-clang/fp16-register.bin
	cmp $(WORK)/x86_64-linux-clang/fp16-reference.bin $(WORK)/x86_64-linux-clang/fp16-register.bin

$(WORK)/x86_64-linux-clang/test_fp16_reference: $(DELTANET_ROOT)/tests/test_fp16_register.cpp $(wildcard $(DELTANET_ROOT)/src/kernels/*.h) $(wildcard $(DELTANET_ROOT)/src/utils/*.h) $(wildcard $(HTP_BUILD_ROOT)/include/*.h) $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -DDELTANET_FP16_REGISTER_REUSE=0 -UNDEBUG -O2 -ffunction-sections -fdata-sections $< -Wl,--gc-sections $(X86_LDFLAGS) -o $@

$(WORK)/x86_64-linux-clang/test_fp16_register: $(DELTANET_ROOT)/tests/test_fp16_register.cpp $(wildcard $(DELTANET_ROOT)/src/kernels/*.h) $(wildcard $(DELTANET_ROOT)/src/utils/*.h) $(wildcard $(HTP_BUILD_ROOT)/include/*.h) $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -DDELTANET_FP16_REGISTER_REUSE=1 -UNDEBUG -O2 -ffunction-sections -fdata-sections $< -Wl,--gc-sections $(X86_LDFLAGS) -o $@

$(WORK)/x86_64-linux-clang/test_reduction: $(DELTANET_ROOT)/tests/test_reduction.cpp $(wildcard $(DELTANET_ROOT)/src/kernels/*.h) $(wildcard $(DELTANET_ROOT)/src/utils/*.h) $(wildcard $(HTP_BUILD_ROOT)/include/*.h) $(WORK)/x86_64-linux-clang/.build-config | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -UDELTANET_U16_KARATSUBA_REDUCTION -UDELTANET_U16_RADIX256_REDUCTION -O2 -ffunction-sections -fdata-sections $< -Wl,--gc-sections $(X86_LDFLAGS) -o $@

$(WORK)/x86_64-linux-clang $(WORK)/aarch64-android $(WORK)/hexagon-v79 $(WORK)/hexagon-v81 $(WORK)/hexagon-v85:
	@mkdir -p $@/ops $@/attention

# Keep the stamp's mtime unchanged when the effective build configuration agrees.
$(WORK)/x86_64-linux-clang/.build-config: FORCE | $(WORK)/x86_64-linux-clang
	@python3 $(HTP_BUILD_ROOT)/write_build_config.py "$@" X86_BUILD_CONFIG

$(WORK)/aarch64-android/.build-config: FORCE | $(WORK)/aarch64-android
	@python3 $(HTP_BUILD_ROOT)/write_build_config.py "$@" AARCH64_BUILD_CONFIG

$(WORK)/x86_64-linux-clang/%.o: $(SRC_DIR)/%.cpp $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/x86_64-linux-clang/ops/%.o: $(OP_SRC_DIR)/%.cpp $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/x86_64-linux-clang/attention/%.o: $(ATTENTION_SRC_DIR)/%.cpp $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(WORK)/x86_64-linux-clang
	@mkdir -p $(@D)
	$(X86_CXX) $(X86_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/x86_64-linux-clang/$(LIBRARY_NAME): $(addprefix $(WORK)/x86_64-linux-clang/,$(OTHER_OBJS) $(OP_OBJS)) $(WORK)/x86_64-linux-clang/.build-config $(RULE_FILES) | $(HFILES)
	$(X86_CXX) -fPIC -std=c++17 -g -shared -o $@ $(filter %.o,$^) $(X86_LDFLAGS)

$(WORK)/aarch64-android/%.o: $(SRC_DIR)/%.cpp $(WORK)/aarch64-android/.build-config $(RULE_FILES) | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(ARM_CLANG_OPTS) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/aarch64-android/ops/%.o: $(OP_SRC_DIR)/%.cpp $(WORK)/aarch64-android/.build-config $(RULE_FILES) | $(WORK)/aarch64-android
	$(AARCH64_CXX) $(ARM_CLANG_OPTS) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/aarch64-android/attention/%.o: $(ATTENTION_SRC_DIR)/%.cpp $(WORK)/aarch64-android/.build-config $(RULE_FILES) | $(WORK)/aarch64-android
	@mkdir -p $(@D)
	$(AARCH64_CXX) $(ARM_CLANG_OPTS) $(AARCH64_CXX_FLAGS) -DTHIS_PKG_NAME=$(PACKAGE_NAME) -MMD -MP -c $< -o $@

$(WORK)/aarch64-android/$(LIBRARY_NAME): $(addprefix $(WORK)/aarch64-android/,$(OTHER_OBJS) $(OP_OBJS)) $(WORK)/aarch64-android/.build-config $(RULE_FILES) | $(HFILES)
	$(AARCH64_CXX) $(ARM_CLANG_OPTS) -fPIC -std=c++17 -g -shared -o $@ $(filter %.o,$^) $(AARCH64_LDFLAGS)

define HEXAGON_RULES
$(WORK)/hexagon-v$(1)/.build-config: FORCE | $(WORK)/hexagon-v$(1)
	@python3 $(HTP_BUILD_ROOT)/write_build_config.py "$$@" HEXAGON_BUILD_CONFIG arch=$(1)

$(WORK)/hexagon-v$(1)/%.o: $(SRC_DIR)/%.cpp $(WORK)/hexagon-v$(1)/.build-config $(RULE_FILES) | $(WORK)/hexagon-v$(1)
	$$(HEXAGON_CXX) $$(HEXAGON_BASE_FLAGS) -mv$(1) -DQCOM_HTP_ARCH=$(1) -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/qurt -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/posix -I$$(HEXAGON_SDK_ROOT)/incs -I$$(HEXAGON_SDK_ROOT)/incs/stddef -DTHIS_PKG_NAME=$$(PACKAGE_NAME) -MMD -MP -c $$< -o $$@

$(WORK)/hexagon-v$(1)/ops/%.o: $(OP_SRC_DIR)/%.cpp $(WORK)/hexagon-v$(1)/.build-config $(RULE_FILES) | $(WORK)/hexagon-v$(1)
	$$(HEXAGON_CXX) $$(HEXAGON_BASE_FLAGS) -mv$(1) -DQCOM_HTP_ARCH=$(1) -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/qurt -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/posix -I$$(HEXAGON_SDK_ROOT)/incs -I$$(HEXAGON_SDK_ROOT)/incs/stddef -DTHIS_PKG_NAME=$$(PACKAGE_NAME) -MMD -MP -c $$< -o $$@

$(WORK)/hexagon-v$(1)/attention/%.o: $(ATTENTION_SRC_DIR)/%.cpp $(WORK)/hexagon-v$(1)/.build-config $(RULE_FILES) | $(WORK)/hexagon-v$(1)
	@mkdir -p $$(@D)
	$$(HEXAGON_CXX) $$(HEXAGON_BASE_FLAGS) -mv$(1) -DQCOM_HTP_ARCH=$(1) -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/qurt -I$$(HEXAGON_SDK_ROOT)/rtos/qurt/computev$(1)/include/posix -I$$(HEXAGON_SDK_ROOT)/incs -I$$(HEXAGON_SDK_ROOT)/incs/stddef -DTHIS_PKG_NAME=$$(PACKAGE_NAME) -MMD -MP -c $$< -o $$@

$(WORK)/hexagon-v$(1)/$(LIBRARY_NAME): $$(addprefix $(WORK)/hexagon-v$(1)/,$$(OTHER_OBJS) $$(OP_OBJS) $$(if $$(filter 81,$(1)),$$(HMX_OBJECTS))) $(WORK)/hexagon-v$(1)/.build-config $(RULE_FILES) | $$(HFILES)
	$$(HEXAGON_CXX) -mv$(1) -fPIC -std=c++17 -g -shared -o $$@ $$(filter %.o,$$^) $$(HEXAGON_LDFLAGS)

$(WORK)/hexagon-v$(1)/libet_qnnop.so: $$(addprefix $(WORK)/hexagon-v$(1)/,$$(OTHER_OBJS) $$(OP_OBJS) $$(if $$(filter 81,$(1)),$$(HMX_OBJECTS))) $$(NATIVE_OBJECTS) $(WORK)/hexagon-v$(1)/.build-config $(RULE_FILES) | $$(HFILES)
	$$(HEXAGON_CXX) -mv$(1) -fPIC -std=c++17 -g -shared -o $$@ $$(filter %.o,$$^) $$(HEXAGON_LDFLAGS)

$(WORK)/hexagon-v$(1)/$(HTP_LIBRARY_NAME): $(WORK)/hexagon-v$(1)/$(LIBRARY_NAME)
	cp $$< $$@
endef

$(eval $(call HEXAGON_RULES,79))
$(eval $(call HEXAGON_RULES,81))
$(eval $(call HEXAGON_RULES,85))

$(WORK)/hexagon-v81/hmx/%.o: $(HMX_ROOT)/src/qhblas_hmx/%.c $(WORK)/hexagon-v81/.build-config $(RULE_FILES)
	@mkdir -p $(@D)
	$(HEXAGON_CC) -O3 -fPIC -mv81 -mhmx -mhvx -mhvx-length=128B -I$(HMX_ROOT)/inc/qhblas_hmx -I$(HEXAGON_SDK_ROOT)/libs/qhl_hvx/inc/internal -I$(HEXAGON_SDK_ROOT)/rtos/qurt/computev81/include/qurt -MMD -MP -c $< -o $@

$(WORK)/hexagon-v81/hmx/%.o: $(HMX_ROOT)/src/qhblas_hmx/%.S $(WORK)/hexagon-v81/.build-config $(RULE_FILES)
	@mkdir -p $(@D)
	$(HEXAGON_CC) -fPIC -mv81 -mhmx -MMD -MP -c $< -o $@

clean:
	-rm -rf $(WORK)

FORCE:

# Objects depend on RULE_FILES. Older dependency files are replaced by that
# rebuild, and may still name headers removed by a source reorganization.
DEP_FILES := $(foreach target,x86_64-linux-clang aarch64-android hexagon-v79 hexagon-v81 hexagon-v85,\
    $(addprefix $(WORK)/$(target)/,$(OTHER_OBJS:.o=.d) $(OP_OBJS:.o=.d)))
-include $(shell for htp_dep in $(wildcard $(DEP_FILES)); do \
    htp_stale=; \
    for htp_rule in $(RULE_FILES); do \
      if test "$$htp_rule" -nt "$$htp_dep"; then htp_stale=1; break; fi; \
    done; \
    if test -z "$$htp_stale"; then printf '%s\n' "$$htp_dep"; fi; \
  done)

.PHONY: all clean FORCE htp_x86 htp_aarch64 htp_v79 htp_v81 htp_v85 test-attention test-attention-u16 test-attention-stream test-hmx-tile-layout test-chunk test-chunk-state-update test-reduction test-reference-reduction test-fp16-register test-op-package-interface
