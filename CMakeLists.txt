add_executable(fluid mimir_fluid.cu)
target_link_libraries(fluid PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(fluid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
