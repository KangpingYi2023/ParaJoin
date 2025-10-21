find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

set(TEST_TARGETS buildindex search search_multi search_join)

foreach(target IN LISTS TEST_TARGETS)
    add_executable(${target} ${target}.cpp)
    # target_include_directories(${target} PRIVATE ${KaHIP_INCLUDE_DIR})
    target_link_libraries(${target} PRIVATE TBB::tbb faiss OpenMP::OpenMP_CXX)
endforeach()


