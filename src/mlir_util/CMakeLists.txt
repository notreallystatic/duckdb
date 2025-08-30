add_library(mlir_util STATIC mlir_util.cpp)

target_link_libraries(mlir_util PUBLIC lingodb_ext ${LIBS})