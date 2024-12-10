add_executable(gol GOL_mimir.cu)
target_link_libraries(gol PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(gol PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
