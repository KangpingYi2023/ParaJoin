#pragma once

#include <faiss/IndexPQ.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <thread>
#include <map>
#include "utils.h"
#include "searcher.hpp"
#include <bitset>

namespace VectorJoin
{
    typedef std::pair<float, int> PFI;

    template <typename dist_t>
    class iRangeGraph_Build
    {
    public:
        size_t max_threads = 32;
        SegmentTree *tree;
        DataLoader *storage;
        std::vector<std::vector<std::vector<std::pair<float, int>>>> edges;
        std::vector<std::vector<PFI>> reverse_edges;
        size_t M;
        size_t ef_construction;

        size_t pq_M, pq_nbits, pq_D, pq_N;
        int centroids_nb;
        faiss::ProductQuantizer *pq_ptr;
        std::vector<int> labels;
        std::vector<float> landmark_distances;
        std::vector<std::vector<float>> all_centroids;
        std::vector<std::vector<int>> candidate_entry_points;

        hnswlib::L2Space *space;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_{nullptr};
        std::vector<size_t> visitedpool;
        size_t visited_tag{0};
        std::mutex tag_mutex;

        std::queue<int> threadidpool;

        iRangeGraph_Build(DataLoader *store, int M_out = 32, int ef_c = 400, int PQ_M = 8, int PQ_nbits = 2) : storage(store), M(M_out), ef_construction(ef_c), pq_M(PQ_M), pq_nbits(PQ_nbits)
        {
            space = new hnswlib::L2Space(storage->Dim);
            fstdistfunc_ = space->get_dist_func();
            dist_func_param_ = space->get_dist_func_param();
            tree = new SegmentTree(storage->data_nb);
            tree->BuildTree(tree->root);
            edges.resize(storage->data_nb);
            for (int i = 0; i < storage->data_nb; i++)
            {
                edges[i].resize(tree->max_depth + 1);
            }
            visitedpool.resize(storage->data_nb);
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            return fstdistfunc_(v1.data(), v2.data(), dist_func_param_);
        }

        void copyfirstchild(TreeNode *u)
        {
            TreeNode *firstchild = u->childs[0];
            for (int id = firstchild->lbound; id <= firstchild->rbound; id++)
            {
                int lower_layer = firstchild->depth;
                int higher_layer = u->depth;
                edges[id][higher_layer] = edges[id][lower_layer];
            }
        }

        std::priority_queue<PFI> search_on_incomplete_graph(TreeNode *u, std::vector<float> &query_point, int ef, int query_k, std::vector<int> enterpoints)
        {
            size_t local_tag;
            {
                std::lock_guard<std::mutex> lock(tag_mutex);
                visited_tag++;
                local_tag = visited_tag;
            }

            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> pool;
            std::priority_queue<PFI> candidates;

            for (auto pid : enterpoints)
            {
                float dis = dis_compute(query_point, storage->data_points[pid]);
                visitedpool[pid] = local_tag;
                pool.emplace(dis, pid);
                candidates.emplace(dis, pid);
            }

            float lowerBound = candidates.top().first;

            int layer = u->depth;

            while (!pool.empty())
            {
                auto current_pair = pool.top();
                if (current_pair.first > lowerBound)
                    break;
                pool.pop();
                int current_pointId = current_pair.second;
                size_t size = edges[current_pointId][layer].size();

                for (int i = 0; i < size; i++)
                {
                    int neighborId = edges[current_pointId][layer][i].second;
                    if (visitedpool[neighborId] == local_tag)
                        continue;
                    visitedpool[neighborId] = local_tag;
                    float dis = dis_compute(query_point, storage->data_points[neighborId]);
                    if (candidates.size() < ef || dis < lowerBound)
                    {
                        candidates.emplace(dis, neighborId);
                        pool.emplace(dis, neighborId);

                        if (candidates.size() > ef)
                        {
                            candidates.pop();
                        }
                        if (candidates.size())
                        {
                            lowerBound = candidates.top().first;
                        }
                    }
                }
            }

            while (candidates.size() > query_k)
                candidates.pop();
            return candidates;
        }

        std::vector<PFI> PruneByHeuristic2(std::vector<PFI> &old_list, std::vector<PFI> &new_list)
        {
            std::priority_queue<PFI, std::vector<PFI>, std::greater<PFI>> queue_closest;
            std::vector<PFI> return_list;
            std::vector<bool> return_list_belong_to_lowerlayer_list;
            for (auto t : old_list)
                queue_closest.emplace(t);
            for (auto t : new_list)
                queue_closest.emplace(t);
            if (queue_closest.size() <= M)
            {
                while (queue_closest.size())
                {
                    return_list.emplace_back(queue_closest.top());
                    queue_closest.pop();
                }
                return return_list;
            }

            while (queue_closest.size())
            {
                if (return_list.size() >= M)
                    break;

                auto current_pair = queue_closest.top();
                float dist_to_pid = current_pair.first;
                queue_closest.pop();

                bool good = true;
                bool current_old = false;
                for (auto t : old_list)
                {
                    if (t.second == current_pair.second)
                    {
                        current_old = true;
                        break;
                    }
                }
                for (int i = 0; i < return_list.size(); i++)
                {
                    if (current_old && return_list_belong_to_lowerlayer_list[i])
                        continue;
                    auto second_pair = return_list[i];
                    float curdist = dis_compute(storage->data_points[current_pair.second], storage->data_points[second_pair.second]);
                    if (curdist < dist_to_pid)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.emplace_back(current_pair);
                    return_list_belong_to_lowerlayer_list.emplace_back(current_old);
                }
            }
            return return_list;
        }

        void process_node(TreeNode *u)
        {
            if (u->childs.size() == 0)
                return;

            copyfirstchild(u);
            int merged_point_num = u->childs[0]->rbound - u->childs[0]->lbound + 1;
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            for (int i = 1; i < u->childs.size(); i++)
            {
                std::uniform_int_distribution<int> u_start(0, merged_point_num - 1);
                TreeNode *cur_child = u->childs[i];
                for (int pid = cur_child->lbound; pid <= cur_child->rbound; pid++)
                {
                    std::vector<int> enterpoints;
                    for (int i = 0; i < std::min(3, merged_point_num); i++)
                    {
                        int enterpid = u_start(e) + u->lbound;
                        enterpoints.emplace_back(enterpid);
                    }

                    auto search_result = search_on_incomplete_graph(u, storage->data_points[pid], ef_construction, ef_construction, enterpoints);
                    while (search_result.size())
                    {
                        edges[pid][u->depth].emplace_back(search_result.top());
                        search_result.pop();
                    }
                    edges[pid][u->depth] = PruneByHeuristic2(edges[pid][cur_child->depth], edges[pid][u->depth]);
                }

                for (int j = 0; j < merged_point_num; j++)
                {
                    int pid = u->lbound + j;
                    reverse_edges[pid].clear();
                }

                for (int pid = cur_child->lbound; pid <= cur_child->rbound; pid++)
                {
                    for (auto neighbor_pair : edges[pid][u->depth])
                    {
                        int neighborId = neighbor_pair.second;
                        if (neighborId < cur_child->lbound)
                        {
                            reverse_edges[neighborId].emplace_back(neighbor_pair.first, pid);
                        }
                    }
                }

                for (int j = 0; j < merged_point_num; j++)
                {
                    int pid = u->lbound + j;
                    edges[pid][u->depth] = PruneByHeuristic2(edges[pid][u->depth], reverse_edges[pid]);
                }

                merged_point_num += cur_child->rbound - cur_child->lbound + 1;
            }
        }

        float dist_compute_naive(std::vector<float> a, std::vector<float> b, int Dim)
        {
            float dist = 0.0f;
            for (int i = 0; i < Dim; i++)
            {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }

            return dist;
        }

        void generatelandmarks()
        {
            pq_D = storage->Dim;
            pq_N = storage->data_nb;
            int pq_K = 1 << pq_nbits;
            centroids_nb = pow(pq_K, pq_M);
            labels.resize(storage->data_nb);
            landmark_distances.resize(storage->data_nb);
            all_centroids.resize(centroids_nb);
            candidate_entry_points.resize(centroids_nb);

            std::vector<std::priority_queue<PFI>> top_entry_points(centroids_nb);

            pq_ptr = new faiss::ProductQuantizer(pq_D, pq_M, pq_nbits);

            std::vector<float> data(pq_N * pq_D);
            for (int i = 0; i < pq_N; i++)
                std::copy(storage->data_points[i].begin(), storage->data_points[i].end(), data.data() + i * pq_D);

            pq_ptr->train(pq_N, data.data());

            size_t code_size = (pq_M * pq_nbits + 1) / 8;

            std::vector<uint8_t> codes(code_size);
            for (int i = 0; i < centroids_nb; ++i)
            {
                int base = 1 << 8;
                all_centroids[i].resize(pq_D);
                int tmp = i;
                for (int m = 0; m < code_size; ++m)
                {
                    codes[m] = tmp % base;
                    tmp /= base;
                }

                pq_ptr->decode(codes.data(), all_centroids[i].data());
            }

            std::vector<uint8_t> data_codes(pq_N * code_size);
            pq_ptr->compute_codes(data.data(), data_codes.data(), pq_N);

            // const float *data_ptr = data.data();
            // const float *centroids_ptr = all_centroids.data();
            for (int i = 0; i < pq_N; i++)
            {

                uint8_t *code_tmp = data_codes.data() + i * code_size;
                int min_idx = 0;
                int base = 1 << 8;

                for (int m = 0; m < code_size; m++)
                {
                    min_idx += code_tmp[m] * static_cast<uint64_t>(std::pow(base, m));
                    // const float *part = pq_ptr->get_centroids(m, code_tmp[m]);
                    // std::copy(part, part + pq_D / pq_M, centroid.data() + m * (pq_D / pq_M));
                }
                std::vector<float> &centroid = all_centroids[min_idx];
                float min_dist = dis_compute(storage->data_points[i], centroid);

                labels[i] = min_idx;
                auto &heap = top_entry_points[min_idx];
                heap.emplace(min_dist, i);
                if (heap.size() > 2 * M)
                    heap.pop();
                landmark_distances[i] = std::sqrt(min_dist);
            }

            for (int i = 0; i < centroids_nb; ++i)
            {
                auto &heap = top_entry_points[i];
                while (!heap.empty())
                {
                    auto dist_pair = heap.top();
                    candidate_entry_points[i].push_back(dist_pair.second);
                    heap.pop();
                }
                std::reverse(candidate_entry_points[i].begin(), candidate_entry_points[i].end());
            }

            std::cout << "Label for point 100: " << labels[100] << ", Distance for point 100: " << landmark_distances[100] << std::endl;
        }

        void generatelandmarks_with_RaBitQ()
        {
            pq_D = storage->Dim;
            pq_N = storage->data_nb;
            int pq_K = 1 << pq_nbits;
            // centroids_nb = pow(pq_K, pq_M);
            labels.resize(storage->data_nb);
            landmark_distances.resize(storage->data_nb);
            all_centroids.clear();

            faiss::RaBitQuantizer pq(pq_D, faiss::METRIC_L2);

            std::vector<float> data(pq_N * pq_D);
            for (int i = 0; i < pq_N; i++)
                std::copy(storage->data_points[i].begin(), storage->data_points[i].end(), data.data() + i * pq_D);

            size_t code_bits = pq_D;
            size_t code_size = pq.code_size;

            std::unordered_map<std::string, std::vector<float>> id_to_centroid;
            std::vector<std::string> raw_id(pq_N);

            std::vector<uint8_t> data_codes(pq_N * code_size);
            pq.compute_codes(data.data(), data_codes.data(), pq_N);

            // const float *data_ptr = data.data();
            // const float *centroids_ptr = all_centroids.data();
            for (int i = 0; i < pq_N; i++)
            {

                uint8_t *code_tmp = data_codes.data() + i * code_size;

                std::string id = std::string((const char *)code_tmp, code_size);

                raw_id[i] = id;

                if (id_to_centroid.find(id) == id_to_centroid.end())
                {
                    std::vector<float> centroid(pq_D);
                    pq.decode(code_tmp, centroid.data(), 1);
                    id_to_centroid[id] = centroid;
                }
            }

            int index = 0;
            std::unordered_map<std::string, int> id_to_index;
            for (const auto &kv : id_to_centroid)
            {
                id_to_index[kv.first] = index++;
                all_centroids.push_back(kv.second); // 顺序与 index 一致
            }

            centroids_nb = index;
            for (int i = 0; i < pq_N; ++i)
            {
                int idx = id_to_index[raw_id[i]];
                labels[i] = idx;

                float dist = dis_compute(storage->data_points[i], all_centroids[idx]);
                landmark_distances[i] = std::sqrt(dist);
            }

            std::cout << "Label for point 100: " << labels[100] << ", Distance for point 100: " << landmark_distances[100] << std::endl;
        }

        void buildindex()
        {
            std::vector<std::vector<TreeNode *>> level_nodes;
            level_nodes.resize(tree->max_depth + 1);
            for (auto node : tree->treenodes)
            {
                level_nodes[node->depth].emplace_back(node);
            }
            for (int layer = tree->max_depth; layer >= 0; layer--)
            {
                std::cout << "building for layer " << layer << std::endl;
                std::vector<std::thread> threads;

                for (int i = 0; i < level_nodes[layer].size(); i++)
                {
                    if (threads.size() >= max_threads)
                    {
                        for (auto &thread : threads)
                        {
                            if (thread.joinable())
                            {
                                thread.join();
                            }
                        }
                        threads.clear();
                    }
                    TreeNode *u = level_nodes[layer][i];
                    threads.emplace_back(std::thread(&iRangeGraph_Build<dist_t>::process_node, this, u));
                }
                for (auto &thread : threads)
                {
                    if (thread.joinable())
                    {
                        thread.join();
                    }
                }

                threads.clear();
            }
        }

        void buildandsave(std::string indexpath)
        {
            if (std::filesystem::exists(indexpath))
            {
                std::cout << "Index for " + indexpath + " already exist! Skip." << std::endl;
            }
            else
            {
                CheckPath(indexpath);
                std::ofstream indexfile(indexpath, std::ios::out | std::ios::binary);
                if (!indexfile.is_open())
                    throw Exception("cannot open " + indexpath);
                reverse_edges.resize(storage->data_nb);
                timeval t1, t2;
                gettimeofday(&t1, NULL);
                buildindex();
                gettimeofday(&t2, NULL);
                double construction_time = GetTime(t1, t2);

                std::string index_logpath = indexpath + "_log.txt";
                CheckPath(index_logpath);
                std::ofstream index_logfile(index_logpath, std::ios::out);
                std::cout << "construction time:" << construction_time << "s" << std::endl;
                index_logfile << "construction time:" << construction_time << "s" << std::endl;

                for (int pid = 0; pid < storage->data_nb; pid++)
                {
                    for (int layer = 0; layer <= tree->max_depth; layer++)
                    {
                        int size = edges[pid][layer].size();
                        indexfile.write((char *)&size, sizeof(int));
                        for (int i = 0; i < size; i++)
                        {
                            int neighborId = edges[pid][layer][i].second;
                            indexfile.write((char *)&neighborId, sizeof(int));
                        }
                    }
                }

                std::cout << "save index done" << std::endl;
                indexfile.close();
            }
        }
    };

}