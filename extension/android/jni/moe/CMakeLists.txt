cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(qwen35_moe_jni LANGUAGES C CXX)
endif()
if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "Qwen35 MoE JNI requires Android arm64-v8a")
endif()
get_filename_component(_repo "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(_moe "${_repo}/examples/qualcomm/oss_scripts/moe/qwen35-35B-A3B")
set(EXECUTORCH_ANDROID_BUILD
    "${_repo}/build-android"
    CACHE PATH "Prebuilt ExecuTorch Android runtime"
)
set(MOE_BUILD_DIR
    "${_repo}/build/moe/qwen35-35B-A3B"
    CACHE PATH "MoE build.sh output"
)
set(QNN_SDK_ROOT
    "$ENV{QNN_SDK_ROOT}"
    CACHE PATH "QNN SDK matching the model export"
)
set(HEXAGON_SDK_ROOT
    "$ENV{HEXAGON_SDK_ROOT}"
    CACHE PATH "Hexagon SDK"
)
foreach(
  _file
  "${MOE_BUILD_DIR}/native-prefill/integer_rpc_stub.o"
  "${MOE_BUILD_DIR}/native-decode/integer_rpc_stub.o"
  "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h"
)
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "Missing ${_file}; build the MoE runtime first")
  endif()
endforeach()
set(_moe_dsp_arch 85)
if(EXISTS "${MOE_BUILD_DIR}/TARGET.json")
  file(READ "${MOE_BUILD_DIR}/TARGET.json" _moe_target)
  string(JSON _moe_dsp_arch GET "${_moe_target}" target dsp_arch)
  string(REGEX REPLACE "^v" "" _moe_dsp_arch "${_moe_dsp_arch}")
endif()
if(NOT _moe_dsp_arch MATCHES "^(79|81|85)$")
  message(FATAL_ERROR "Unsupported MoE DSP architecture: ${_moe_dsp_arch}")
endif()
file(GLOB _prefill CONFIGURE_DEPENDS "${_moe}/runner/prefill/*.cpp")
file(GLOB _decode CONFIGURE_DEPENDS "${_moe}/runner/decode/*.cpp")
file(GLOB _common CONFIGURE_DEPENDS "${_moe}/runner/*.cpp")
file(GLOB _chat CONFIGURE_DEPENDS "${_moe}/runner/chat/*.cpp")
file(GLOB _absl "${EXECUTORCH_ANDROID_BUILD}/lib/libabsl_*.a")
add_library(
  qwen35_moe_jni SHARED
  "${CMAKE_CURRENT_LIST_DIR}/../jni_layer_qwen35_moe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session.cpp"
  ${_prefill}
  ${_common}
  ${_chat}
  "${_repo}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/qwen_image.cpp"
  "${_moe}/runner/shared/graph_binding.cpp"
  "${MOE_BUILD_DIR}/native-prefill/integer_rpc_stub.o"
)
target_compile_features(qwen35_moe_jni PRIVATE cxx_std_17)
target_compile_options(
  qwen35_moe_jni PRIVATE -O3 -fno-emulated-tls -fvisibility=hidden
                         -fvisibility-inlines-hidden
)
target_compile_definitions(
  qwen35_moe_jni
  PRIVATE MOE_CHAT_RUNNER MOE_JNI_RUNNER C10_USING_CUSTOM_GENERATED_MACROS
          SM_MOE_DSP_ARCH=${_moe_dsp_arch}
          ET_EVENT_TRACER_ENABLED ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0
          ET_ENABLE_PROGRAM_VERIFICATION=0
)
target_include_directories(
  qwen35_moe_jni
  PRIVATE "${_repo}/.."
          "${_repo}/third-party/flatbuffers/include"
          "${EXECUTORCH_ANDROID_BUILD}/schema/include"
          "${_repo}/runtime/core/portable_type/c10"
          "${EXECUTORCH_ANDROID_BUILD}/include"
          "${MOE_BUILD_DIR}/native-prefill"
          "${HEXAGON_SDK_ROOT}/incs"
          "${HEXAGON_SDK_ROOT}/incs/stddef"
          "${QNN_SDK_ROOT}/include/QNN"
)
find_library(
  _qwen35_cdsprpc cdsprpc
  PATHS "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64"
  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED
)
target_link_options(
  qwen35_moe_jni PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
  "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/../version_script.txt"
)
set(
  _runtime_libraries
  "${EXECUTORCH_ANDROID_BUILD}/libexecutorch.a"
  "${EXECUTORCH_ANDROID_BUILD}/libexecutorch_core.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/module/libextension_module.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/data_loader/libextension_data_loader.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/tensor/libextension_tensor.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/flat_tensor/libextension_flat_tensor.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/named_data_map/libextension_named_data_map.a"
)
target_link_libraries(
  qwen35_moe_jni
  PRIVATE
    -Wl,--whole-archive
    "${EXECUTORCH_ANDROID_BUILD}/lib/libregex_lookahead.a"
    -Wl,--no-whole-archive
    -Wl,--start-group
    ${_runtime_libraries}
    "${EXECUTORCH_ANDROID_BUILD}/lib/libtokenizers.a"
    "${EXECUTORCH_ANDROID_BUILD}/lib/libsentencepiece.a"
    "${EXECUTORCH_ANDROID_BUILD}/lib/libre2.a"
    "${EXECUTORCH_ANDROID_BUILD}/lib/libpcre2-8.a"
    "${EXECUTORCH_ANDROID_BUILD}/lib/libextension_llm_sampler.a"
    ${_absl}
    -Wl,--end-group
    ${_qwen35_cdsprpc}
    dl
    log
    z
)
install(TARGETS qwen35_moe_jni LIBRARY DESTINATION jniLibs/arm64-v8a)

add_library(
  qwen35_moe_decode SHARED
  ${_decode}
  ${_common}
  "${_moe}/runner/shared/graph_binding.cpp"
  "${MOE_BUILD_DIR}/native-decode/integer_rpc_stub.o"
)
set_target_properties(qwen35_moe_decode PROPERTIES OUTPUT_NAME moe_decode)
target_compile_features(qwen35_moe_decode PRIVATE cxx_std_17)
target_compile_options(
  qwen35_moe_decode PRIVATE -O3 -fvisibility=hidden -fvisibility-inlines-hidden
)
target_compile_definitions(
  qwen35_moe_decode
  PRIVATE MOE_DECODE_LIBRARY C10_USING_CUSTOM_GENERATED_MACROS
          SM_MOE_DSP_ARCH=${_moe_dsp_arch}
          ET_EVENT_TRACER_ENABLED ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0
          ET_ENABLE_PROGRAM_VERIFICATION=0
)
target_include_directories(
  qwen35_moe_decode
  PRIVATE "${_repo}/.."
          "${_repo}/third-party/flatbuffers/include"
          "${EXECUTORCH_ANDROID_BUILD}/schema/include"
          "${_repo}/runtime/core/portable_type/c10"
          "${EXECUTORCH_ANDROID_BUILD}/include"
          "${MOE_BUILD_DIR}/native-decode"
          "${HEXAGON_SDK_ROOT}/incs"
          "${HEXAGON_SDK_ROOT}/incs/stddef"
          "${QNN_SDK_ROOT}/include/QNN"
)
target_link_options(
  qwen35_moe_decode PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
  "-Wl,-rpath,$ORIGIN/../prefill"
  "-Wl,--version-script=${_moe}/runner/decode.exports"
)
target_link_libraries(
  qwen35_moe_decode PRIVATE -Wl,--start-group ${_runtime_libraries}
                           -Wl,--end-group ${_qwen35_cdsprpc} dl log z
)
install(TARGETS qwen35_moe_decode LIBRARY DESTINATION jniLibs/arm64-v8a)
install(
  FILES "${MOE_BUILD_DIR}/native-prefill/libinteger_rpc_skel.so"
  DESTINATION jniLibs/arm64-v8a
  RENAME libqwen35_prefill_skel.so
)
install(
  FILES "${MOE_BUILD_DIR}/native-decode/libinteger_rpc_skel.so"
  DESTINATION jniLibs/arm64-v8a
  RENAME libqwen35_decode_skel.so
)

# Its lifetime fix must accompany the new JNI DSO, rather than copying an older
# terminal-runner provider that retained a QNN device after model close.
add_library(
  qwen35_moe_shared SHARED
  "${_moe}/runner/shared/context_store.cpp"
  "${_moe}/runner/shared/weight_source.cpp"
  "${_moe}/runner/shared/graph_binding.cpp"
  "${_moe}/runner/decode/qnn_front.cpp"
  "${_moe}/runner/runtime_utils.cpp"
)
set_target_properties(qwen35_moe_shared PROPERTIES OUTPUT_NAME qwen35_shared)
target_compile_features(qwen35_moe_shared PRIVATE cxx_std_17)
target_compile_options(
  qwen35_moe_shared PRIVATE -O3 -fvisibility=hidden -fvisibility-inlines-hidden
)
target_include_directories(
  qwen35_moe_shared PRIVATE "${QNN_SDK_ROOT}/include/QNN"
)
target_link_options(
  qwen35_moe_shared PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
)
target_link_libraries(qwen35_moe_shared PRIVATE dl)
install(TARGETS qwen35_moe_shared LIBRARY DESTINATION jniLibs/arm64-v8a)
set(MOE_OP_PACKAGE
    "${MOE_BUILD_DIR}/op-package/hexagon-v${_moe_dsp_arch}/libQnnDeltaNetOpPackage_HTP.so"
    CACHE FILEPATH "GDN op package matching the Native MoE build"
)
install(FILES "${MOE_OP_PACKAGE}" DESTINATION jniLibs/arm64-v8a)
