#pragma once
#pragma unroll(4)

#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <unordered_map>
#include <omp.h>
#include <tbb/tbb.h>
#include <tbb/concurrent_hash_map.h>
#include <algorithm>
#include "utils.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>

namespace VectorJoin
{
    typedef std::pair<tableint, tableint> KeyPair;

    struct PairHash
    {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const
        {
            auto h1 = std::hash<tableint>{}(p.first);
            auto h2 = std::hash<tableint>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    template <typename dist_t>
    class ParaJoin
    {
    public:
        int query_nb, M;
        float partition_time;
        tbb::spin_mutex mtx;
        std::atomic<int> metric_threads{0};
        std::atomic<long long> metric_hop{0};
        std::atomic<long long> metric_distance_computation{0};
        JoinDataLoader *join_storage;
        iRangeGraph_Search<dist_t> *graph_A;
        iRangeGraph_Search<dist_t> *graph_B;
        int max_threads;
        tbb::concurrent_unordered_map<int, std::vector<tableint>> tmp_edge_A;
        tbb::concurrent_unordered_map<int, std::vector<tableint>> tmp_edge_B;
        tbb::concurrent_unordered_map<std::pair<tableint, tableint>, float, PairHash> dist_cache;
        tbb::concurrent_unordered_map<tableint, tbb::concurrent_vector<tableint>> reverse_cache;
        tbb::global_control *thread_limiter;
        searcher::Bitset<uint64_t> *cache_contain_set = nullptr;
        float lambda_threshold, dis_threshold;
        std::string join_type;
        std::ofstream log_file;

        ParaJoin(iRangeGraph_Search<dist_t> *graphA,
                           iRangeGraph_Search<dist_t> *graphB,
                           int m, JoinDataLoader *j_storage, std::string j_type, std::string log_path,
                           float l_threshold = 0.0f, float d_threshold = 0.0f, int m_threads = 32)
            : graph_A(graphA), graph_B(graphB), M(m), join_storage(j_storage), join_type(j_type), lambda_threshold(l_threshold), dis_threshold(d_threshold), max_threads(m_threads)
        {
            tmp_edge_A.reserve(graph_A->max_elements_);
            tmp_edge_B.reserve(graph_B->max_elements_);
            dist_cache.reserve(graph_A->max_elements_ * graph_B->max_elements_);
            reverse_cache.reserve(std::max(graph_A->max_elements_, graph_B->max_elements_));
            log_file.open(log_path, std::ios::out | std::ios::trunc);
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, max_threads);
            omp_set_num_threads(max_threads);
        }

        ~ParaJoin() {}

        float dist_compute_naive(float *a, float *b, int Dim)
        {
            float dist = 0.0f;
            for (int i = 0; i < Dim; i++)
            {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }

            return std::sqrt(dist);
        }

        std::vector<tableint> calculate_cardinality(int ql, int qr, iRangeGraph_Search<dist_t> *graph, size_t &car)
        {
            std::vector<tableint> res;
            res.reserve(graph->max_elements_);

            std::vector<TreeNode *> filterednodes = graph->tree->filter_and_estimation(car, graph->tree->root, ql, qr);

            for (auto u : filterednodes)
                for (int i = u->lbound; i <= u->rbound; i++)
                    res.push_back(i);

            return res;
        }

        std::vector<tableint> calculate_cardinality_naive(int ql, int qr, iRangeGraph_Search<dist_t> *graph, size_t &car)
        {
            std::vector<tableint> res;
            res.reserve(graph->max_elements_);

            for (size_t i = 0; i < graph->max_elements_; i++)
            {
                if (i >= ql && i <= qr)
                {
                    res.push_back(i);
                    car++;
                }
            }

            return res;
        }

        // AkNN search
        std::vector<tableint> aknn_search(tableint query_id,
                                          const void *query_data,
                                          iRangeGraph_Search<dist_t> *target_graph,
                                          std::vector<tableint> &target_vectors,
                                          tbb::concurrent_unordered_map<int, std::vector<tableint>> *tmp_edges,
                                          //   std::vector<std::vector<tableint>> *tmp_entries,
                                          std::vector<tableint> entry_points,
                                          //   std::vector<std::vector<tableint>> *tmp_edges,
                                          int ql, int qr, int k, int ef, bool cached = false)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            // unsigned seed = 1234;
            std::default_random_engine e(seed);

            long long local_dist_compute = 0;
            long long local_hop = 0;
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            std::priority_queue<PFI> top_candidates;
            searcher::Bitset<uint64_t> visited_set(target_graph->max_elements_);
            std::uniform_int_distribution<int> u_start(0, target_vectors.size() - 1);

            if (entry_points.empty())
            {
                std::priority_queue<PFI> entry_heap;
                for (int i = 0; i < k; i++)
                {
                    int pid;
                    do
                    {
                        int idx = u_start(e); // Random start point
                        pid = target_vectors[idx];
                    } while (visited_set.get(pid));

                    visited_set.set(pid);
                    char *ep_data = target_graph->getDataByInternalId(pid);
                    float dis = target_graph->fstdistfunc_(query_data, ep_data, target_graph->dist_func_param_);

                    local_dist_compute++;
                    entry_heap.emplace(dis, pid);
                }

                while (!entry_heap.empty())
                {
                    entry_points.push_back(entry_heap.top().second);
                    entry_heap.pop();
                }
            }

            for (auto pid : entry_points)
            {
                visited_set.set(pid);
                char *ep_data = target_graph->getDataByInternalId(pid);
                float dis = target_graph->fstdistfunc_(query_data, ep_data, target_graph->dist_func_param_);
                local_dist_compute++;

                candidate_set.emplace(dis, pid);
                top_candidates.emplace(dis, pid);
            }

            float lowerBound = top_candidates.top().first;

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                if (current_point_pair.first > lowerBound)
                    break;

                candidate_set.pop();
                int current_pid = current_point_pair.second;

                std::vector<tableint> &selected_edges = (*tmp_edges)[current_pid];

                int num_edges = selected_edges.size();

                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(target_graph->getDataByInternalId(selected_edges[i]), target_graph->prefetch_lines);
                }

                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];

                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);

                    float dis;
                    char *neighbor_data = target_graph->getDataByInternalId(neighbor_id);
                    dis = target_graph->fstdistfunc_(query_data, neighbor_data, target_graph->dist_func_param_);
                    local_dist_compute++;

                    if (top_candidates.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        lowerBound = top_candidates.top().first;
                    }
                    else if (dis < lowerBound)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        top_candidates.emplace(dis, neighbor_id);
                        top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
            }

            while (top_candidates.size() > k)
                top_candidates.pop();

            std::vector<tableint> aknn_results;
            while (!top_candidates.empty())
            {
                auto result_pair = top_candidates.top();
                aknn_results.push_back(result_pair.second);
                top_candidates.pop();
            }

            metric_distance_computation.fetch_add(local_dist_compute, std::memory_order_relaxed);
            metric_hop.fetch_add(local_hop, std::memory_order_relaxed);

            return aknn_results;
        }

        std::vector<tableint> range_search(const void *query_data,
                                           iRangeGraph_Search<dist_t> *target_graph,
                                           tbb::concurrent_unordered_map<int, std::vector<tableint>> *tmp_edges,
                                           std::vector<tableint> entry_points,
                                           std::vector<tableint> &target_vectors,
                                           int ql, int qr, float threshold, int ef)
        {
            long long local_dist_compute = 0;
            long long local_hop = 0;
            std::vector<tableint> range_results;
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> candidate_set;
            searcher::Bitset<uint64_t> visited_set(target_graph->max_elements_);

            if (!entry_points.empty())
            {
                for (auto pid : entry_points)
                {
                    visited_set.set(pid);
                    char *ep_data = target_graph->getDataByInternalId(pid);
                    float dis = target_graph->fstdistfunc_(query_data, ep_data, target_graph->dist_func_param_);
                    local_dist_compute++;

                    candidate_set.emplace(dis, pid);
                    if (dis < threshold)
                    {
                        range_results.push_back(pid);
                    }
                }
            }

            if (candidate_set.empty())
            {
                // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
                unsigned seed = 1234;
                std::default_random_engine e(seed);

                int num_vectors = target_vectors.size();
                std::uniform_int_distribution<int> u_start(0, num_vectors - 1);
                for (int i = 0; i < std::min(5, num_vectors); i++)
                {
                    int pid;
                    do
                    {
                        int idx = u_start(e); // Random start point
                        pid = target_vectors[idx];
                    } while (visited_set.get(pid));

                    visited_set.set(pid);
                    char *ep_data = target_graph->getDataByInternalId(pid);
                    float dis = target_graph->fstdistfunc_(query_data, ep_data, target_graph->dist_func_param_);
                    local_dist_compute++;
                    candidate_set.emplace(dis, pid);
                    if (dis < threshold)
                    {
                        range_results.push_back(pid);
                    }
                }
            }

            while (!candidate_set.empty())
            {
                auto current_point_pair = candidate_set.top();
                // ++metric_hops;
                if (candidate_set.size() >= ef && current_point_pair.first > threshold)
                    break;

                candidate_set.pop();
                int current_pid = current_point_pair.second;

                std::vector<tableint> &selected_edges = (*tmp_edges)[current_pid];
                // std::vector<tableint> selected_edges = target_graph->SelectEdge(current_pid, ql, qr, M, visited_set);

                int num_edges = selected_edges.size();
                for (int i = 0; i < std::min(num_edges, 3); ++i)
                {
                    memory::mem_prefetch_L1(target_graph->getDataByInternalId(selected_edges[i]), target_graph->prefetch_lines);
                }

                for (int i = 0; i < num_edges; ++i)
                {
                    int neighbor_id = selected_edges[i];

                    if (visited_set.get(neighbor_id))
                        continue;
                    visited_set.set(neighbor_id);
                    char *neighbor_data = target_graph->getDataByInternalId(neighbor_id);
                    float dis = target_graph->fstdistfunc_(query_data, neighbor_data, target_graph->dist_func_param_);
                    local_dist_compute++;

                    if (dis < threshold)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                        range_results.push_back(neighbor_id);
                    }
                    else if (candidate_set.size() < ef)
                    {
                        candidate_set.emplace(dis, neighbor_id);
                    }
                }
            }

            metric_distance_computation.fetch_add(local_dist_compute, std::memory_order_relaxed);
            metric_hop.fetch_add(local_hop, std::memory_order_relaxed);

            return range_results;
        }

        // 执行Loop Join
        std::vector<KeyPair> loop_join(
            std::vector<tableint> &outer_vectors,
            std::vector<tableint> &inner_vectors,
            iRangeGraph_Search<dist_t> *outer_graph,
            iRangeGraph_Search<dist_t> *inner_graph,
            int outer_ql, int outer_qr,
            int inner_ql, int inner_qr,
            int k, int ef, bool swap_flag)
        {
            // std::cout << "Execting Loop join!" << std::endl;
            tbb::concurrent_vector<KeyPair> results;

            auto *outer_tmp_edges = swap_flag ? &tmp_edge_B : &tmp_edge_A;
            auto *inner_tmp_edges = swap_flag ? &tmp_edge_A : &tmp_edge_B;

            tbb::concurrent_unordered_map<tableint, std::vector<tableint>> inner_aknn_cache;
            tbb::parallel_for_each(outer_vectors.begin(), outer_vectors.end(),
                                   [&](tableint vo)
                                   {
                                       auto ak_vo = aknn_search(vo, outer_graph->getDataByInternalId(vo), inner_graph, inner_vectors, inner_tmp_edges, inner_ql, inner_qr, k, ef, false);
                                       for (tableint vi : ak_vo)
                                       {
                                           //    std::vector<tableint> ak_vi;
                                           auto &ak_vi = inner_aknn_cache[vi];
                                           if (ak_vi.empty())
                                               ak_vi = aknn_search(vi, inner_graph->getDataByInternalId(vi), outer_graph, outer_vectors, outer_tmp_edges, outer_ql, outer_qr, k, ef, true);

                                           if (std::find(ak_vi.begin(), ak_vi.end(), vo) != ak_vi.end())
                                               results.emplace_back(swap_flag ? std::pair(vi, vo) : std::pair(vo, vi));
                                       }
                                   });

            std::vector<KeyPair> final_results(results.begin(), results.end());

            return final_results;
        }

        // execute Merge Join operator
        std::vector<KeyPair> merge_join(
            std::vector<tableint> &outer_vectors,
            std::vector<tableint> &inner_vectors,
            iRangeGraph_Search<dist_t> *outer_graph,
            iRangeGraph_Search<dist_t> *inner_graph,
            int outer_ql, int outer_qr,
            int inner_ql, int inner_qr,
            int k, int ef, bool swap_flag)
        {
            log_file << "Execting Merge join!" << std::endl;

            auto *outer_tmp_edges = swap_flag ? &tmp_edge_B : &tmp_edge_A;
            auto *inner_tmp_edges = swap_flag ? &tmp_edge_A : &tmp_edge_B;

            timeval t1, t2;
            gettimeofday(&t1, NULL);
            tbb::tick_count tb0 = tbb::tick_count::now();
            tbb::concurrent_unordered_map<tableint, std::vector<tableint>> outer_aknn_map;
            tbb::concurrent_vector<KeyPair> concurrent_result;

            tbb::task_arena arena(max_threads);
            arena.execute([&]
                          { tbb::parallel_for_each(outer_vectors.begin(), outer_vectors.end(), [&](tableint vo)
                                                   {
                                       auto &res = outer_aknn_map[vo];
                                       if (res.empty())
                                       {
                                           res = aknn_search(vo, outer_graph->getDataByInternalId(vo), inner_graph, inner_vectors, inner_tmp_edges, {}, inner_ql, inner_qr, k, ef, false);
                                       } });

                tbb::tick_count tb1 = tbb::tick_count::now();
                log_file << "AKNN search time for Table A(tbb analysis): " << (tb1 - tb0).seconds() << " seconds" << std::endl;
                gettimeofday(&t2, NULL);
                auto duration = GetTime(t1, t2);
                log_file << "AKNN search time for Table A: " << duration << std::endl;

                gettimeofday(&t1, NULL);
                tbb::parallel_for(outer_aknn_map.range(),
                                  [&](const auto &range)
                                  {
                                      for (const auto &pair : range)
                                      {
                                          const tableint key = pair.first;
                                          const std::vector<tableint> &values = pair.second;

                                          // add (value, key) to reverse knn cache
                                          for (const tableint value : values)
                                          {
                                              reverse_cache[value].push_back(key);
                                          }
                                      }
                                  });

                gettimeofday(&t2, NULL);
                duration = GetTime(t1, t2);
                log_file << "Cache time for Table A: " << duration << std::endl;

                gettimeofday(&t1, NULL);
                tbb::concurrent_unordered_map<tableint, std::vector<tableint>> inner_aknn_map;
                tbb::parallel_for_each(inner_vectors.begin(), inner_vectors.end(),
                                       [&](tableint vi)
                                       {
                                           auto &res = inner_aknn_map[vi];
                                           if (res.empty())
                                           {
                                               std::vector<tableint> entry_points;
                                               if (reverse_cache.find(vi) != reverse_cache.end())
                                               {
                                                   for (auto pid : reverse_cache[vi])
                                                       entry_points.push_back(pid);
                                               }
                                               res = aknn_search(vi, inner_graph->getDataByInternalId(vi), outer_graph, outer_vectors, outer_tmp_edges, entry_points, outer_ql, outer_qr, k, ef, true);
                                           }
                                       });

                gettimeofday(&t2, NULL);
                duration = GetTime(t1, t2);
                log_file << "AKNN search time for Table B: " << duration << std::endl;
                gettimeofday(&t1, NULL);
                // 3) join
                
                tbb::parallel_for_each(outer_vectors.begin(), outer_vectors.end(),
                                       [&](tableint vo)
                                       {
                                           for (tableint vi : outer_aknn_map[vo])
                                               if (std::find(inner_aknn_map[vi].begin(), inner_aknn_map[vi].end(), vo) != inner_aknn_map[vi].end())
                                                   concurrent_result.emplace_back(swap_flag ? std::pair(vi, vo) : std::pair(vo, vi));
                                       });
                gettimeofday(&t2, NULL);
                duration = GetTime(t1, t2);
                log_file << "result generating time: " << duration << std::endl; });

            std::vector<KeyPair> results(concurrent_result.begin(), concurrent_result.end());

            return results;
        }

        std::vector<std::vector<KeyPair>> graph_index_partition(
            std::vector<tableint> &target_vectors,
            tbb::concurrent_unordered_map<int, std::vector<tableint>> *target_tmp_edges,
            // std::vector<KeyPair> &partitions,
            // std::vector<unsigned int> &partition_offsets,
            int num_partitions)
        {
            int n = target_vectors.size();
            int part_size_limit = (n + num_partitions - 1) / num_partitions;
            std::vector<std::vector<KeyPair>> partitions;

            tableint max_id = target_vectors.back();
            searcher::Bitset<uint64_t> visited_set(max_id + 1);

            for (int start = 0; start < n; start++)
            {
                tableint start_id = target_vectors[start];
                if (visited_set.get(start_id))
                    continue;

                std::vector<KeyPair> part;
                std::stack<KeyPair> candidates;
                auto start_pair = std::make_pair(start_id, start_id);
                candidates.push(start_pair);

                while (part.size() < part_size_limit && !candidates.empty())
                {
                    auto top_pair = candidates.top();
                    auto current_id = top_pair.second;
                    candidates.pop();

                    if (visited_set.get(current_id))
                        continue;

                    part.push_back(top_pair);
                    visited_set.set(current_id);

                    auto neighbors = (*target_tmp_edges)[current_id];

                    for (auto neighbor : neighbors)
                    {
                        if (!visited_set.get(neighbor))
                        {
                            auto neighbor_pair = std::make_pair(current_id, neighbor);
                            candidates.push(neighbor_pair);
                        }
                    }
                }

                partitions.push_back(part);
            }

            return partitions;
        }

        void graph_reconstruction(int A_ql, int A_qr, int B_ql, int B_qr,
                                  std::vector<tableint> &A_vectors,
                                  std::vector<tableint> &B_vectors)
        {
            tmp_edge_A.clear();
            tmp_edge_B.clear();
            reverse_cache.clear();

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, A_vectors.size()),
                [&](const tbb::blocked_range<size_t> &range)
                {
                    for (size_t i = range.begin(); i < range.end(); ++i)
                    {
                        tableint current_pid = A_vectors[i];
                        auto new_edges = graph_A->SelectEdgeParallel(current_pid, A_ql, A_qr, M);
                        tmp_edge_A[current_pid] = new_edges;
                    }
                },
                tbb::simple_partitioner() 
            );

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, B_vectors.size()),
                [&](const tbb::blocked_range<size_t> &range)
                {
                    for (size_t i = range.begin(); i < range.end(); ++i)
                    {
                        tableint current_pid = B_vectors[i];
                        auto new_edges = graph_B->SelectEdgeParallel(current_pid, B_ql, B_qr, M);
                        tmp_edge_B[current_pid] = new_edges;
                    }
                },
                tbb::simple_partitioner() 
            );
        }

        // execute SPJ(range join)
        std::vector<KeyPair> execute_mknnjoin_query(
            int A_ql, int A_qr,
            int B_ql, int B_qr,
            int k = 10,
            int ef = 100)
        { 
            timeval t1, t2;
            gettimeofday(&t1, NULL);
            // Calculate cardinalities
            size_t C_A = 0, C_B = 0;
            std::vector<tableint> outer_vectors = calculate_cardinality(A_ql, A_qr, graph_A, C_A);
            std::vector<tableint> inner_vectors = calculate_cardinality(B_ql, B_qr, graph_B, C_B);

            graph_reconstruction(A_ql, A_qr, B_ql, B_qr, outer_vectors, inner_vectors);

            iRangeGraph_Search<dist_t> *outer_graph = graph_A;
            iRangeGraph_Search<dist_t> *inner_graph = graph_B;
            int outer_ql = A_ql, outer_qr = A_qr;
            int inner_ql = B_ql, inner_qr = B_qr;

            bool swap_flag = false;
            if (C_A > C_B)
            {
                std::swap(outer_graph, inner_graph);
                std::swap(outer_ql, inner_ql);
                std::swap(outer_qr, inner_qr);
                std::swap(outer_vectors, inner_vectors);
                swap_flag = true;
            }

            float lambda = static_cast<float>(inner_vectors.size()) /
                           static_cast<float>(outer_vectors.size());

            gettimeofday(&t2, NULL);
            auto duration = GetTime(t1, t2);
            log_file << "Before join time: " << duration << std::endl;

            return merge_join(outer_vectors, inner_vectors,
                              outer_graph, inner_graph,
                              outer_ql, outer_qr,
                              inner_ql, inner_qr,
                              k, ef, swap_flag);
        }

        std::vector<KeyPair> execute_rangejoin_query(
            int A_ql, int A_qr,
            int B_ql, int B_qr,
            float threshold = 10000.0f,
            int ef = 100, 
            int beta = 16)
        {
            // Calculate cardinalities
            size_t C_A = 0, C_B = 0;
            std::vector<tableint> outer_vectors = calculate_cardinality(A_ql, A_qr, graph_A, C_A);
            std::vector<tableint> inner_vectors = calculate_cardinality(B_ql, B_qr, graph_B, C_B);

            graph_reconstruction(A_ql, A_qr, B_ql, B_qr, outer_vectors, inner_vectors);

            // Choose outer table and inner table.
            iRangeGraph_Search<dist_t> *outer_graph = graph_A;
            iRangeGraph_Search<dist_t> *inner_graph = graph_B;
            int outer_ql = A_ql, outer_qr = A_qr;
            int inner_ql = B_ql, inner_qr = B_qr;
            auto *outer_tmp_edges = &tmp_edge_A;
            auto *inner_tmp_edges = &tmp_edge_B;

            bool swap_flag = false;
            if (C_A > C_B)
            {
                std::swap(outer_graph, inner_graph);
                std::swap(outer_ql, inner_ql);
                std::swap(outer_qr, inner_qr);
                std::swap(outer_vectors, inner_vectors);
                std::swap(outer_tmp_edges, inner_tmp_edges);
                swap_flag = true;
            }

            tbb::concurrent_vector<KeyPair> results;

            auto max_id = outer_vectors.back();
            std::vector<std::vector<tableint>> outer_cache(max_id + 1);
            
            timeval t1, t2;
            partition_time = 0;
            gettimeofday(&t1, NULL);
            auto partitions = graph_index_partition(outer_vectors, outer_tmp_edges, max_threads * beta);
            gettimeofday(&t2, NULL);
            partition_time = GetTime(t1, t2);
            
            tbb::task_arena arena(max_threads);
            arena.execute([&]{
                tbb::parallel_for_each(partitions.begin(), partitions.end(), [&](std::vector<KeyPair> partition)
                                   {
                for (int i = 0; i < partition.size(); i++)
                {
                    auto current_pair = partition[i];
                    auto pre_vo = current_pair.first;
                    auto vo = current_pair.second;

                    std::vector<tableint> entry_points;
                    if (vo != pre_vo)
                    {
                        entry_points = outer_cache[pre_vo];
                    }
                    auto res = range_search(outer_graph->getDataByInternalId(vo), inner_graph, inner_tmp_edges, entry_points, inner_vectors, inner_ql, inner_qr, threshold, ef);
                    outer_cache[vo]=res;

                    for (tableint vi : res)
                        results.emplace_back(swap_flag ? std::pair(vi, vo) : std::pair(vo, vi));
                } });
            });

            std::vector<KeyPair> final_results(results.begin(), results.end());

            return final_results;
        }

        void search(std::vector<int> &SearchEF, std::string saveprefix, std::string join_type, int K = 0, float threshold = 0.0f)
        {
            std::vector<int> Beta_list={16};
            tbb::global_control global_controller(tbb::global_control::max_allowed_parallelism, max_threads);
            for (auto pair : join_storage->join_range)
            {
                int suffix = pair.first;
                const auto &ranges = pair.second;
                std::vector<std::vector<KeyPair>> &gt = join_storage->groundtruth[suffix];

#ifdef RELEASE_BUILD
                std::string savepath = saveprefix + "join_" + std::to_string(suffix) + ".csv";
                CheckPath(savepath);
                std::ofstream outfile(savepath, std::ios::app);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
#endif

                std::vector<float> HOP;
                std::vector<float> DCO;
                std::vector<float> QPS;
                std::vector<float> RECALL;
                std::vector<float> APR;
                std::vector<float> AQT;
                std::vector<int> BETA;
                std::vector<float> PT;

                for (auto beta : Beta_list){
                    for (auto ef : SearchEF)
                    {
                        int tp = 0;
                        float searchtime = 0;
                        float total_partition_time = 0;

                        metric_hop.store(0);
                        metric_distance_computation.store(0);

                        std::cout << "join type: " << join_type << ", suffix = " << suffix << ", ef = " << ef << ", threads = " << max_threads << ", beta = " << beta << ", threads = " << max_threads << std::endl;
                        log_file << "join type: " << join_type << ", suffix = " << suffix << ", ef = " << ef << ", beta = " << beta << std::endl;
                        for (int i = 0; i < join_storage->query_nb; i++)
                        {
                            const auto &rp = ranges[i];
                            std::pair<int, int> r1 = rp.first;
                            std::pair<int, int> r2 = rp.second;

                            timeval t1, t2;
                            std::vector<KeyPair> res;
                            if (join_type == "mknn")
                            {
                                gettimeofday(&t1, NULL);
                                res = execute_mknnjoin_query(r1.first, r1.second, r2.first, r2.second, K, ef);
                                gettimeofday(&t2, NULL);
                            }
                            else if (join_type == "range")
                            {
                                gettimeofday(&t1, NULL);
                                res = execute_rangejoin_query(r1.first, r1.second, r2.first, r2.second, threshold, ef, beta);
                                gettimeofday(&t2, NULL);
                            }
                            else
                            {
                                throw std::invalid_argument("Unknown join type " + join_type + "!");
                            }
                            auto duration = GetTime(t1, t2);
                            searchtime += duration;
                            total_partition_time += partition_time;
                            std::stringstream ss;
                            ss << "Query " << i << " in join_" << suffix << " finished! Tuple numbers: " << res.size()
                            << ", search time " << duration << ", dist computations: " << metric_distance_computation.load() << "\n";
                            std::cout << ss.str();
                            log_file << ss.str();
                            log_file.flush();

#ifdef RELEASE_BUILD
                        auto gt_sorted = gt[i];
                        std::sort(gt_sorted.begin(), gt_sorted.end());

                        tp += tbb::parallel_reduce(
                            tbb::blocked_range<size_t>(0, res.size()),
                            0,
                            [&](const tbb::blocked_range<size_t> &r, size_t local_tp) -> size_t
                            {
                                for (size_t j = r.begin(); j < r.end(); ++j)
                                {
                                    const auto &x_pair = res[j];
                                    if (std::binary_search(gt_sorted.begin(), gt_sorted.end(), x_pair))
                                        local_tp++;
                                }
                                return local_tp;
                            },
                            std::plus<size_t>());
#endif
                    }

                        long long metric_distance_computations = metric_distance_computation.load();
                        long long metric_hops = metric_hop.load();
                        std::cout << "Groundtruth number: " << join_storage->groundtruth_number[suffix] << std::endl;
                        float recall = 1.0 * tp / join_storage->groundtruth_number[suffix];
                        float qps = join_storage->query_nb / searchtime;
                        float aqt = searchtime / join_storage->query_nb;
                        float dco = metric_distance_computations * 1.0 / join_storage->query_nb;
                        float hop = metric_hops * 1.0 / join_storage->query_nb;
                        float apr = hop / dco;
                        float pt = total_partition_time / join_storage->query_nb;

                        HOP.emplace_back(hop);
                        DCO.emplace_back(dco);
                        QPS.emplace_back(qps);
                        RECALL.emplace_back(recall);
                        APR.emplace_back(apr);
                        AQT.emplace_back(aqt);
                        BETA.emplace_back(beta);
                        PT.emplace_back(pt);
                    }
                }
                

#ifdef RELEASE_BUILD
                for (int i = 0; i < RECALL.size(); i++)
                {
                    outfile << SearchEF[i % SearchEF.size()] << "," << RECALL[i] << "," << QPS[i] << "," << AQT[i] << "," << DCO[i] << "," << HOP[i] << "," << APR[i] << "," << BETA[i] << "," << PT[i] << std::endl;
                }
                outfile.close();
                std::cout << "Saved results to " << savepath << std::endl;
#endif
            }
        }
    };
}