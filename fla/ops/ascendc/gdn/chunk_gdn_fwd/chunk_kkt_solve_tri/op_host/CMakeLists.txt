add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kkt_solve_tri_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kkt_solve_tri ACLNNTYPE aclnn_exclude)
add_ops_compile_options(
    OP_NAME ChunkKktSolveTri
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/gdn_preprocess/chunk_scaled_dot_kkt/op_kernel
            -I${PROJECT_SOURCE_DIR}/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel
)
