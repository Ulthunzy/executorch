cmake_minimum_required(VERSION 3.24)
project(qualcomm_moe LANGUAGES C CXX)
get_filename_component(MOE_REPO "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "QNN SDK")
set(HEXAGON_SDK_ROOT "$ENV{HEXAGON_SDK_ROOT}" CACHE PATH "Hexagon SDK")
set(EXECUTORCH_ANDROID_BUILD "${MOE_REPO}/build-android" CACHE PATH "ExecuTorch Android build")
set(MOE_NATIVE_BUILD "${CMAKE_BINARY_DIR}" CACHE PATH "Generated FastRPC interfaces and DSP library")
set(MOE_DSP_ARCH "81" CACHE STRING "HTP ISA: 79, 81 or 85")
option(MOE_BUILD_TOOLS "Build CLI runners and the export capture tool" ON)
if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "MoE requires Android arm64-v8a")
endif()
if(NOT MOE_DSP_ARCH MATCHES "^(79|81|85)$")
  message(FATAL_ERROR "Unsupported HTP ISA")
endif()
include(runner/native/validate_build.cmake)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${MOE_NATIVE_BUILD}/NATIVE_TARGET.json")
add_subdirectory(runner)
file(GLOB _absl "${EXECUTORCH_ANDROID_BUILD}/lib/libabsl_*.a")
set(_et_libraries
  "${EXECUTORCH_ANDROID_BUILD}/libexecutorch.a"
  "${EXECUTORCH_ANDROID_BUILD}/libexecutorch_core.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/module/libextension_module.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/data_loader/libextension_data_loader.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/tensor/libextension_tensor.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/flat_tensor/libextension_flat_tensor.a"
  "${EXECUTORCH_ANDROID_BUILD}/extension/named_data_map/libextension_named_data_map.a")
find_library(_rpc cdsprpc PATHS "${HEXAGON_SDK_ROOT}/ipc/fastrpc/remote/ship/android_aarch64" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
add_library(et_infer SHARED
  "${MOE_NATIVE_BUILD}/native-prefill/et_prefill_rpc_stub.o"
  "${MOE_NATIVE_BUILD}/native-decode/et_decode_rpc_stub.o"
  "${MOE_REPO}/extension/android/jni/moe/jni_layer.cpp"
  "${MOE_REPO}/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/qwen_image.cpp")
target_link_libraries(et_infer PRIVATE moe_chat_runtime moe_decode_runtime moe_context_store)
target_compile_features(et_infer PRIVATE cxx_std_17)
target_compile_options(et_infer PRIVATE -O3 -fno-emulated-tls -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
target_compile_definitions(et_infer PRIVATE MOE_CHAT_RUNNER MOE_JNI_RUNNER MOE_DECODE_LIBRARY
  C10_USING_CUSTOM_GENERATED_MACROS SM_MOE_DSP_ARCH=${MOE_DSP_ARCH} ET_EVENT_TRACER_ENABLED
  ET_ENABLE_DEPRECATED_CONSTANT_BUFFER=0 ET_ENABLE_PROGRAM_VERIFICATION=0)
target_include_directories(et_infer PRIVATE "${MOE_REPO}/.."
  "${MOE_REPO}/runtime/core/portable_type/c10" "${MOE_REPO}/third-party/flatbuffers/include"
  "${EXECUTORCH_ANDROID_BUILD}/include" "${EXECUTORCH_ANDROID_BUILD}/schema/include"
  "${MOE_NATIVE_BUILD}/native-prefill" "${MOE_NATIVE_BUILD}/native-decode"
  "${HEXAGON_SDK_ROOT}/incs" "${HEXAGON_SDK_ROOT}/incs/stddef" "${QNN_SDK_ROOT}/include/QNN")
target_link_options(et_infer PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL -Wl,--gc-sections
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/runner/infer.exports")
set_property(TARGET et_infer APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runner/infer.exports")
target_link_libraries(et_infer PRIVATE -Wl,--whole-archive
  "${EXECUTORCH_ANDROID_BUILD}/lib/libregex_lookahead.a" -Wl,--no-whole-archive
  -Wl,--start-group ${_et_libraries} "${EXECUTORCH_ANDROID_BUILD}/lib/libtokenizers.a"
  "${EXECUTORCH_ANDROID_BUILD}/lib/libsentencepiece.a" "${EXECUTORCH_ANDROID_BUILD}/lib/libre2.a"
  "${EXECUTORCH_ANDROID_BUILD}/lib/libpcre2-8.a" "${EXECUTORCH_ANDROID_BUILD}/lib/libextension_llm_sampler.a"
  ${_absl} -Wl,--end-group ${_rpc} dl log z)
install(CODE [[
  foreach(_legacy libmoe_decode.so libqwen35_shared.so libqwen35_moe_jni.so
      libqwen35_prefill_skel.so libqwen35_decode_skel.so libQnnDeltaNetOpPackage_HTP.so)
    file(REMOVE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/jniLibs/arm64-v8a/${_legacy}")
  endforeach()
]])
install(TARGETS et_infer LIBRARY DESTINATION jniLibs/arm64-v8a)
install(FILES "${MOE_NATIVE_BUILD}/libet_qnnop.so" DESTINATION jniLibs/arm64-v8a)
install(FILES runner/infer_api.h DESTINATION include/et_infer)
if(MOE_BUILD_TOOLS)
  foreach(_runner qwen35_moe_runner prefill_handoff_runner)
    add_executable(${_runner} runner/main.cpp)
    target_link_libraries(${_runner} PRIVATE et_infer)
    target_compile_features(${_runner} PRIVATE cxx_std_17)
    set_target_properties(${_runner} PROPERTIES BUILD_RPATH "$ORIGIN")
  endforeach()
  target_compile_definitions(prefill_handoff_runner PRIVATE MOE_DIAGNOSTIC_RUNNER)
  add_executable(capture_weights tools/capture_weights.cpp)
  target_link_libraries(capture_weights PRIVATE moe_context_store dl)
  target_compile_features(capture_weights PRIVATE cxx_std_17)
  target_include_directories(capture_weights PRIVATE "${QNN_SDK_ROOT}/include/QNN")
endif()
