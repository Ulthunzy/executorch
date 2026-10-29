export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.51.200.261021  #2.45.0.260326 #2.42.0.251225 #2.40.0.251030 #2.39.0.250926 #2.31.0.250130 #2.37.0.250724  # 2.36.3.250722 #2.28.2.241116
# export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/automotive-ai-sdk-for-qualcomm-neural-networks/qnn_sdk_v2.40.0_auto_lnx/qaisw-v2.40.0.251030114326_189385-auto-lnx
export HEXAGON_SDK_ROOT=/hunter/qnn_sdk/hexagon/6.7.0.0
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/21.0.04
export ANDROID_SDK=/home/hunter/Android/Sdk
export ANDROID_NDK=$ANDROID_NDK_ROOT
export LD_LIBRARY_PATH=$HEXAGON_TOOLS_ROOT/Tools/lib:$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}
export PYTHONPATH=".."
# Preserve the caller's CUDA_VISIBLE_DEVICES selection for GPU calibration.
export EXECUTORCH_ROOT=/hunter/cpp_code/ET/npu/executorch
export PYTHONPATH=/hunter/cpp_code/ET/npu/executorch:$PYTHONPATH
export FLATC_EXECUTABLE=$EXECUTORCH_ROOT/data/bin/flatc
export PATH="$EXECUTORCH_ROOT/data/bin:$PATH"
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
