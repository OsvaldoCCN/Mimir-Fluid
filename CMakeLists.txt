add_executable(fluid fluid_simulation.cu)
target_link_libraries(fluid PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(fluid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)