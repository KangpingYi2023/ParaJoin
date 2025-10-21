#pragma once
#pragma unroll(4)

#include <milvus/MilvusClient.h>
#include <vector>
#include <queue>
#include <unordered_map>
// #include <tbb/tbb.h>
// #include <tbb/concurrent_hash_map.h>
#include <shared_mutex>
#include <algorithm>
#include <execution>
#include "utils.h"
#include "searcher.hpp"
#include "memory.hpp"
#include <bitset>

namespace VectorJoin
{
    typedef std::pair<tableint, tableint> KeyPair;

    template <typename dist_t>
    class SPJ_QueryProcessor_Milvus
    {
    public:
        int query_nb, M;
        std::string table_A, table_B;
        std::atomic<long long> metric_hop{0};
        std::atomic<long long> metric_distance_computation{0};
        JoinDataLoader *join_storage;

        float lambda_threshold, dis_threshold;
        std::string join_type;
        std::ofstream log_file;

        SPJ_QueryProcessor_Milvus(std::string t_A, std::string t_B, JoinDataLoader *j_storage, std::string j_type, std::string log_path,
                                  float l_threshold = 0.0f, float d_threshold = 0.0f, int max_threads = 32, std::string host = "127.0.0.1", std::string port = "19530")
            : table_A(t_A), table_B(t_B), join_storage(j_storage), join_type(j_type), lambda_threshold(l_threshold), dis_threshold(d_threshold), client_(MilvusClient::Create())
        {
            ConnectParam param{host, port};
            auto status = client_->Connect(param);
            if (!status.IsOk())
            {
                throw std::runtime_error("Failed to connect to Milvus: " + status.Message());
            }
            log_file.open(log_path, std::ios::out | std::ios::trunc);
            tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, max_threads);
        }

        ~SPJ_QueryProcessor_Milvus() {}

        std::vector<tableint> calculate_cardinality(int domain, int ql, int qr, size_t &car)
        {
            std::vector<tableint> res;
            res.reserve(graph->max_elements_);

            for (int idx = 0; idx < domain; idx++)
            {
                if (idx >= ql && idx <= qr)
                {
                    res.push_back(idx);
                    car++;
                }
            }

            return res;
        }

        std::vector<std::vector<KeyPair>> aknn_search(
            const std::string &collection,
            const std::vector<tableint> &target_vectors,
            int k,
            int ef)
        {
            SearchParam param;
            param.collection_name = collection;
            param.topk = k;
            param.params = {{"ef", ef}};

            std::vector<std::string> partitions; // 可选分区
            std::vector<std::vector<KeyPair>> results(query_vectors.size());

            auto start = std::chrono::high_resolution_clock::now();

            auto status = client_->Search(param, query_vectors, partitions, [&, top_k](int64_t query_index, const QueryResult &res)
                                          {
            for (size_t i = 0; i < res.results.size(); i++) {
                results[query_index].push_back({(int)query_index, (int)res.results[i].id});
            } });

            if (!status.IsOk())
            {
                throw std::runtime_error("Batch search failed: " + status.Message());
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Batch search finished, time = "
                      << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

            return results;
        }

        // execute Merge Join operator
        std::vector<KeyPair> merge_join(
            std::string &outer_tablename,
            std::string &inner_tablename,
            std::vector<tableint> &outer_vectors,
            std::vector<tableint> &inner_vectors,
            int outer_ql, int outer_qr,
            int inner_ql, int inner_qr,
            int k, int ef, bool swap_flag)
        {
            log_file << "Execting Merge join!" << std::endl;
            std::vector<KeyPair> results;

            return results;
        }

        // execute SPJ(range join)
        std::vector<KeyPair> execute_mknnjoin_query(
            int A_ql, int A_qr,
            int B_ql, int B_qr,
            int k = 10,
            int ef = 100)
        {
            size_t C_A = 0, C_B = 0;
            std::string outer_tablename = table_A;
            std::string inner_tablename = table_B;
            std::vector<tableint> outer_vectors = calculate_cardinality(join_storage->storage1->data_nb, A_ql, A_qr, C_A);
            std::vector<tableint> inner_vectors = calculate_cardinality(join_storage->storage2->data_nb, B_ql, B_qr, C_B);

            int outer_ql = A_ql, outer_qr = A_qr;
            int inner_ql = B_ql, inner_qr = B_qr;

            bool swap_flag = false;
            if (C_A > C_B)
            {
                std::swap(outer_tablename, inner_tablename);
                std::swap(outer_ql, inner_ql);
                std::swap(outer_qr, inner_qr);
                std::swap(outer_vectors, inner_vectors);
                swap_flag = true;
            }

            return merge_join(outer_tablename, inner_tablename,
                              outer_vectors, inner_vectors,
                              outer_ql, outer_qr,
                              inner_ql, inner_qr,
                              k, ef, swap_flag);
        }

        std::vector<KeyPair> execute_rangejoin_query(
            int A_ql, int A_qr,
            int B_ql, int B_qr,
            float threshold = 10000.0f,
            int ef = 100)
        {

            return final_results;
        }

        void search(std::vector<int> &SearchEF, std::string saveprefix, std::string join_type, int K = 0, float threshold = 0.0f)
        {
            std::cout << "Search based on Milvus" << std::endl;
            for (auto pair : join_storage->join_range)
            {
                int suffix = pair.first;
                const auto &ranges = pair.second;
                std::vector<std::vector<KeyPair>> &gt = join_storage->groundtruth[suffix];

#ifdef RELEASE_BUILD
                std::string savepath = saveprefix + "join_" + std::to_string(suffix) + ".csv";
                CheckPath(savepath);
                std::ofstream outfile(savepath);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
#endif

                std::vector<float> HOP;
                std::vector<float> DCO;
                std::vector<float> QPS;
                std::vector<float> RECALL;
                std::vector<float> APR;
                std::vector<float> AQT;

                for (auto ef : SearchEF)
                {
                    int tp = 0;
                    float searchtime = 0;

                    metric_hop.store(0);
                    metric_distance_computation.store(0);

                    std::cout << "join type: " << join_type << ", suffix = " << suffix << ", ef = " << ef << std::endl;
                    log_file << "join type: " << join_type << ", suffix = " << suffix << ", ef = " << ef << std::endl;
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
                            res = execute_rangejoin_query(r1.first, r1.second, r2.first, r2.second, threshold, ef);
                            gettimeofday(&t2, NULL);
                        }
                        else
                        {
                            throw std::invalid_argument("Unknown join type " + join_type + "!");
                        }
                        auto duration = GetTime(t1, t2);
                        searchtime += duration;
                        std::stringstream ss;
                        ss << "Query " << i << " in join_" << suffix << " finished! Tuple numbers: " << res.size()
                           << ", search time " << duration << ", dist computations: " << metric_distance_computation.load() << "\n";
                        std::cout << ss.str();
                        log_file << ss.str();
                        log_file.flush();
                        // std::map<KeyPair, int> record;
                        // for (auto x_pair : res)
                        // {
                        //     // if (record.count(x_pair))
                        //     //     throw Exception("repetitive search results");
                        //     // record[x_pair] = 1;
                        //     if (std::find(gt[i].begin(), gt[i].end(), x_pair) != gt[i].end())
                        //         tp++;
                        // }

                        // std::unordered_set<KeyPair, PairHash> record_set;

#ifdef RELEASE_BUILD
                        tp += tbb::parallel_reduce(
                            tbb::blocked_range<size_t>(0, res.size()),
                            0,
                            [&](const tbb::blocked_range<size_t> &r, int local_tp) -> int
                            {
                                for (size_t j = r.begin(); j < r.end(); ++j)
                                {
                                    const auto &x_pair = res[j];
                                    // if (!record_set.insert(x_pair).second)
                                    //     throw Exception("repetitive search results");
                                    if (std::find(gt[i].begin(), gt[i].end(), x_pair) != gt[i].end())
                                    {
                                        local_tp++;
                                    }
                                }
                                return local_tp;
                            },
                            std::plus<int>());
#endif
                    }

                    long long metric_distance_computations = metric_distance_computation.load();
                    long long metric_hops = metric_hop.load();
                    std::cout << "Groundtruth number: " << join_storage->groundtruth_number[suffix] << std::endl;
                    float recall = 1.0 * tp / join_storage->groundtruth_number[suffix];
                    float qps = join_storage->query_nb / searchtime;
                    float dco = metric_distance_computations * 1.0 / join_storage->query_nb;
                    float hop = metric_hops * 1.0 / join_storage->query_nb;
                    float apr = hop / dco;

                    HOP.emplace_back(hop);
                    DCO.emplace_back(dco);
                    QPS.emplace_back(qps);
                    RECALL.emplace_back(recall);
                    APR.emplace_back(apr);
                    AQT.emplace_back(searchtime);
                }

#ifdef RELEASE_BUILD
                for (int i = 0; i < RECALL.size(); i++)
                {
                    outfile << SearchEF[i] << "," << RECALL[i] << "," << QPS[i] << "," << AQT[i] << "," << DCO[i] << "," << HOP[i] << "," << APR[i] << std::endl;
                }
                outfile.close();
                std::cout << "Saved results to " << savepath << std::endl;
#endif
            }
        }
    };
}