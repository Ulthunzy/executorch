add_library(moe_qnn_runtime INTERFACE)
target_sources(moe_qnn_runtime INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/qnn/qnn_front.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qnn/binary_weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/runtime_utils.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qnn/shared/graph_binding.cpp")

add_library(moe_runtime INTERFACE)
target_link_libraries(moe_runtime INTERFACE moe_qnn_runtime)
target_sources(moe_runtime INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/state_manager.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model_package.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/embedding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/handoff.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/resident_handoff.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model_adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qwen35_adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/chat/settings.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/chat/model_session.cpp")

file(GLOB _moe_prefill CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/prefill/*.cpp")
file(GLOB _moe_decode CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/decode/*.cpp")
file(GLOB _moe_chat CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/chat/*.cpp")
list(REMOVE_ITEM _moe_chat "${CMAKE_CURRENT_LIST_DIR}/chat/settings.cpp" "${CMAKE_CURRENT_LIST_DIR}/chat/model_session.cpp")
add_library(moe_prefill_runtime INTERFACE)
target_link_libraries(moe_prefill_runtime INTERFACE moe_runtime)
target_sources(moe_prefill_runtime INTERFACE ${_moe_prefill} "${CMAKE_CURRENT_LIST_DIR}/prompt_processor.cpp")
add_library(moe_decode_runtime INTERFACE)
target_link_libraries(moe_decode_runtime INTERFACE moe_runtime)
target_sources(moe_decode_runtime INTERFACE ${_moe_decode} "${CMAKE_CURRENT_LIST_DIR}/token_generator.cpp" "${CMAKE_CURRENT_LIST_DIR}/expert_manager.cpp")
add_library(moe_chat_runtime INTERFACE)
target_link_libraries(moe_chat_runtime INTERFACE moe_prefill_runtime)
target_sources(moe_chat_runtime INTERFACE ${_moe_chat} "${CMAKE_CURRENT_LIST_DIR}/runner.cpp")
add_library(moe_context_store INTERFACE)
target_link_libraries(moe_context_store INTERFACE moe_qnn_runtime)
target_sources(moe_context_store INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/qnn/shared/context_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/qnn/shared/weight_source.cpp")
