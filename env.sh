export ANDROID_NDK_ROOT=/hunter/android-ndk/android-ndk-r28c
export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/2.45.0.260326  #2.45.0.260326 #2.42.0.251225 #2.40.0.251030 #2.39.0.250926 #2.31.0.250130 #2.37.0.250724  # 2.36.3.250722 #2.28.2.241116
# export QNN_SDK_ROOT=/hunter/qnn_sdk/src/backends/qnn/automotive-ai-sdk-for-qualcomm-neural-networks/qnn_sdk_v2.40.0_auto_lnx/qaisw-v2.40.0.251030114326_189385-auto-lnx
export ANDROID_SDK=/home/hunter/Android/Sdk
export ANDROID_NDK=$ANDROID_NDK_ROOT
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}
export PYTHONPATH=".."
export CUDA_VISIBLE_DEVICES=""
export EXECUTORCH_ROOT=/hunter/cpp_code/ET/npu/executorch
export PYTHONPATH=/hunter/cpp_code/ET/npu/executorch:$PYTHONPATH
export FLATC_EXECUTABLE=$EXECUTORCH_ROOT/data/bin/flatc
export PATH="$EXECUTORCH_ROOT/data/bin:$PATH"
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 all_proxy=socks5://127.0.0.1:7897
