#pragma once
#pragma unroll(4)

#include "space_l2.h"
#include <filesystem>
#include <string>
#include <tbb/tbb.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <map>

class Exception : public std::runtime_error
{
public:
    Exception(const std::string &msg) : std::runtime_error(msg) {}
};

void CheckPath(std::string filename)
{
    std::filesystem::path pathObj(filename);
    std::filesystem::path dirPath = pathObj.parent_path();
    if (!std::filesystem::exists(dirPath))
    {
        try
        {
            if (std::filesystem::create_directories(dirPath))
            {
                std::cout << "Directory created: " << dirPath << std::endl;
            }
            else
            {
                std::cerr << "Failed to create directory: " << dirPath << std::endl;
            }
        }
        catch (std::filesystem::filesystem_error &e)
        {
            throw Exception(e.what());
        }
    }
}

float GetTime(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC;
}

namespace VectorJoin
{
    typedef std::pair<float, int> PFI;
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    class DataLoader
    {
    public:
        int Dim, query_nb, query_K;
        std::vector<std::vector<float>> query_points;
        int data_nb;
        std::vector<std::vector<float>> data_points;
        std::unordered_map<int, std::vector<std::pair<int, int>>> query_range;
        std::unordered_map<int, std::vector<std::vector<int>>> groundtruth;

        DataLoader() {}
        ~DataLoader() {}

        // query vector filename format: 4 bytes: query number; 4 bytes: dimension; query_nb*Dim vectors
        void LoadQuery(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&query_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            query_points.resize(query_nb);
            for (int i = 0; i < query_nb; i++)
            {
                query_points[i].resize(Dim);
                infile.read((char *)query_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // Used only when computing groundtruth and constructing index. Do not use this to load data for search process
        void LoadData(std::string filename)
        {
            std::ifstream infile(filename, std::ios::in | std::ios::binary);
            if (!infile.is_open())
                throw Exception("cannot open " + filename);
            infile.read((char *)&data_nb, sizeof(int));
            infile.read((char *)&Dim, sizeof(int));
            data_points.resize(data_nb);
            for (int i = 0; i < data_nb; i++)
            {
                data_points[i].resize(Dim);
                infile.read((char *)data_points[i].data(), Dim * sizeof(float));
            }
            infile.close();
        }

        // By default generation, 0.bin~9.bin denotes 2^0~2^-9 range fractions, 17.bin denotes mixed range fraction.
        // Before reading the query ranges, make sure query vectors have been read.
        void LoadQueryRange(std::string fileprefix)
        {
            std::vector<int> s;
            for (int i = 0; i < 10; i++)
                s.emplace_back(i);
            s.emplace_back(17);
            for (auto suffix : s)
            {
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql, qr;
                    infile.read((char *)&ql, sizeof(int));
                    infile.read((char *)&qr, sizeof(int));
                    query_range[suffix].emplace_back(ql, qr);
                }
                infile.close();
            }
        }

        // 0.bin~9.bin correspond to groundtruth for 2^0~2^-9 range fractions, 17.bin for mixed fraction
        void LoadGroundtruth(std::string fileprefix)
        {
            for (auto t : query_range)
            {
                int suffix = t.first;
                std::string filename = fileprefix + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                groundtruth[suffix].resize(query_nb);
                for (int i = 0; i < query_nb; i++)
                {
                    groundtruth[suffix][i].resize(query_K);
                    infile.read((char *)groundtruth[suffix][i].data(), query_K * sizeof(int));
                }
                infile.close();
            }
        }
    };

    class QueryGenerator
    {
    public:
        int data_nb, query_nb;
        hnswlib::L2Space *space;

        QueryGenerator(int data_num, int query_num) : data_nb(data_num), query_nb(query_num) {}
        ~QueryGenerator() {}

        void GenerateRange(std::string saveprefix)
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine e(seed);

            std::vector<std::pair<int, int>> rs;
            int current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                if (current_len < 10)
                    throw Exception("dataset size is too small, increase the amount of data objects!");
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            for (auto t : rs)
            {
                int len = t.first, suffix = t.second;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::cout << "save query range to" << savepath << std::endl;
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::uniform_int_distribution<int> u_start(0, data_nb - len);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
                outfile.close();
            }

            rs.clear();
            current_len = data_nb;
            for (int i = 0; i < 10; i++)
            {
                rs.emplace_back(current_len, i);
                current_len /= 2;
            }
            std::string savepath = saveprefix + "17.bin";
            CheckPath(savepath);
            std::cout << "save query range to" << savepath << std::endl;
            std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            for (auto t : rs)
            {
                int len = t.first;
                std::uniform_int_distribution<int> u_start(0, data_nb - len);

                for (int i = 0; i < query_nb / 10; i++)
                {
                    int ql = u_start(e);
                    int qr = ql + len - 1;
                    if (ql >= data_nb || qr >= data_nb)
                        throw Exception("Query range out of bound");
                    outfile.write((char *)&ql, sizeof(int));
                    outfile.write((char *)&qr, sizeof(int));
                }
            }
            outfile.close();
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_((char *)v1.data(), (char *)v2.data(), space->get_dist_func_param());
            return dis;
        }

        void GenerateGroundtruth(std::string saveprefix, DataLoader &storage)
        {
            space = new hnswlib::L2Space(storage.Dim);
            for (auto t : storage.query_range)
            {
                int suffix = t.first;
                std::string savepath = saveprefix + std::to_string(suffix) + ".bin";
                CheckPath(savepath);
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);
                std::cout << "generating for " << t.first << std::endl;
                for (int i = 0; i < query_nb; i++)
                {
                    auto rp = t.second[i];
                    int ql = rp.first, qr = rp.second;
                    std::priority_queue<std::pair<float, int>> ans;
                    for (int j = ql; j <= qr; j++)
                    {
                        float dis = dis_compute(storage.query_points[i], storage.data_points[j]);
                        ans.emplace(dis, j);
                        if (ans.size() > storage.query_K)
                            ans.pop();
                    }
                    while (ans.size())
                    {
                        auto id = ans.top().second;
                        ans.pop();
                        outfile.write((char *)&id, sizeof(int));
                    }
                }
                outfile.close();
            }
        }
    };

    class JoinDataLoader
    {
        typedef unsigned int tableint;

    public:
        int Dim, query_nb, query_K;
        size_t query_modes;
        DataLoader *storage1, *storage2;
        std::unordered_map<int, std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>> join_range;
        std::unordered_map<int, std::vector<std::vector<std::pair<tableint, tableint>>>> groundtruth;
        std::unordered_map<int, int> groundtruth_number;

        JoinDataLoader(DataLoader *store1, DataLoader *store2, int q_nb, int q_modes) : storage1(store1), storage2(store2), query_nb(q_nb), query_modes(q_modes)
        {
            Dim = store1->Dim;
        }
        ~JoinDataLoader() {}

        void LoadJoinRange(std::string fileprefix, std::string query_type = "merge")
        {
            std::vector<int> s;

            // For MergeJoin test
            if (query_type == "merge")
            {
                for (int i = 2; i < 8; i += 2)
                    s.emplace_back(i);
                // s.emplace_back(17);
            }

            // For Debug
            if (query_type == "loop")
            {
                for (int i = 0; i < 10; i++)
                    s.emplace_back(i);
                s.emplace_back(17);
            }

            for (auto suffix : s)
            {
                std::string filename = fileprefix + "join_" + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql1, ql2, qr1, qr2;
                    infile.read((char *)&ql1, sizeof(int));
                    infile.read((char *)&qr1, sizeof(int));
                    infile.read((char *)&ql2, sizeof(int));
                    infile.read((char *)&qr2, sizeof(int));
                    auto r1 = std::make_pair(ql1, qr1);
                    auto r2 = std::make_pair(ql2, qr2);
                    join_range[suffix].emplace_back(r1, r2);
                }
                infile.close();
            }
        }

        void LoadGroundtruth(std::string fileprefix)
        {
            for (auto t : join_range)
            {
                int suffix = t.first;
                std::string filename = fileprefix + "join_" + std::to_string(suffix) + ".bin";
                std::ifstream infile(filename, std::ios::in | std::ios::binary);
                if (!infile.is_open())
                    throw Exception("cannot open " + filename);

                groundtruth[suffix].resize(query_nb);
                groundtruth_number[suffix] = 0;
                for (int i = 0; i < query_nb; i++)
                {
                    int ans_nb;
                    infile.read((char *)&ans_nb, sizeof(int));
                    groundtruth_number[suffix] += ans_nb;
                    groundtruth[suffix][i].resize(ans_nb);
                    infile.read((char *)groundtruth[suffix][i].data(), ans_nb * sizeof(std::pair<tableint, tableint>));
                }
                infile.close();
            }
        }
    };

    class JoinGenerator
    {
    public:
        int data1_nb, data2_nb, query_nb;
        size_t query_modes;
        hnswlib::L2Space *space;
        std::atomic<long long> metric_gt_distances{0};

        JoinGenerator(int data_num1, int data_num2, int query_num, int query_md) : data1_nb(data_num1), data2_nb(data_num2), query_nb(query_num), query_modes(query_md) {}
        ~JoinGenerator() {}

        void GenerateJoinRange(std::string saveprefix, std::string query_type = "merge")
        {
            // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            unsigned seed = 1234;
            std::default_random_engine e(seed);

            std::vector<std::pair<std::pair<int, int>, int>> rs;
            int current_len1 = data1_nb;
            int current_len2 = data2_nb;
            for (int i = 0; i < query_modes; i++)
            {
                if (current_len1 < 10 || current_len2 < 10)
                    throw Exception("dataset size is too small, increase the amount of data objects!");
                current_len1 /= 2;
                if (query_type == "merge")
                    current_len2 /= 2; // Mergejoin
                else if (query_type == "loop")
                    current_len2 = data2_nb / std::pow(2, 10 - i); // Loopjoin
                else
                    throw Exception("unknown join type: " + query_type);

                auto r = std::make_pair(current_len1, current_len2);
                rs.emplace_back(r, i);
            }
            for (auto t : rs)
            {
                int suffix = t.second;
                std::string savepath = saveprefix + "join_" + std::to_string(suffix) + ".bin";
                if (std::filesystem::exists(savepath))
                {
                    std::cout << "Join_ranges for " + savepath + " already exist! Skip." << std::endl;
                    continue;
                }
                CheckPath(savepath);
                std::cout << "save join range to" << savepath << std::endl;
                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);

                int len1 = t.first.first;
                int len2 = t.first.second;
                std::uniform_int_distribution<int> u1_start(0, data1_nb - len1);
                std::uniform_int_distribution<int> u2_start(0, data2_nb - len2);
                for (int i = 0; i < query_nb; i++)
                {
                    int ql1 = u1_start(e);
                    int ql2 = u2_start(e);
                    int qr1 = ql1 + len1 - 1;
                    int qr2 = ql2 + len2 - 1;
                    if (ql1 >= data1_nb || qr1 >= data1_nb || ql2 >= data2_nb || qr2 >= data2_nb)
                        throw Exception("Query range out of bound");

                    outfile.write((char *)&ql1, sizeof(int));
                    outfile.write((char *)&qr1, sizeof(int));
                    outfile.write((char *)&ql2, sizeof(int));
                    outfile.write((char *)&qr2, sizeof(int));
                }
                outfile.close();
            }

            rs.clear();
            current_len1 = data1_nb;
            current_len2 = data2_nb;
            for (int i = 0; i < query_modes; i++)
            {
                auto r = std::make_pair(current_len1, current_len2);
                rs.emplace_back(r, i);
                current_len1 /= 2;
                if (query_type == "merge")
                    current_len2 /= 2; // Mergejoin
                else if (query_type == "loop")
                    current_len2 = data2_nb / std::pow(2, 10 - i); // Loopjoin
                else
                    throw Exception("unknown join type: " + query_type);
            }
            std::string savepath = saveprefix + "join_17.bin";
            if (std::filesystem::exists(savepath))
            {
                std::cout << "Join_ranges for " + savepath + " already exist! Skip." << std::endl;
                return;
            }
            CheckPath(savepath);
            std::cout << "save join range to" << savepath << std::endl;
            std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
            if (!outfile.is_open())
                throw Exception("cannot open " + savepath);

            for (auto t : rs)
            {
                int len1 = t.first.first;
                int len2 = t.first.second;
                std::uniform_int_distribution<int> u1_start(0, data1_nb - len1);
                std::uniform_int_distribution<int> u2_start(0, data2_nb - len2);

                for (int i = 0; i < query_nb / query_modes; i++)
                {
                    int ql1 = u1_start(e);
                    int ql2 = u2_start(e);
                    int qr1 = ql1 + len1 - 1;
                    int qr2 = ql2 + len2 - 1;

                    if (ql1 >= data1_nb || qr1 >= data1_nb || ql2 >= data2_nb || qr2 >= data2_nb)
                        throw Exception("Query range out of bound");

                    outfile.write((char *)&ql1, sizeof(int));
                    outfile.write((char *)&qr1, sizeof(int));
                    outfile.write((char *)&ql2, sizeof(int));
                    outfile.write((char *)&qr2, sizeof(int));
                }
            }
            outfile.close();
        }

        float dis_compute(std::vector<float> &v1, std::vector<float> &v2)
        {
            hnswlib::DISTFUNC<float> fstdistfunc_ = space->get_dist_func();
            float dis = fstdistfunc_(v1.data(), v2.data(), space->get_dist_func_param());
            return dis;
        }

        void GenerateJoinGroundtruth(std::string saveprefix, JoinDataLoader &join_storage, std::string join_type, float dis_threshold = 0.0f)
        {
            space = new hnswlib::L2Space(join_storage.Dim);

            std::string logpath = saveprefix + "gt_record.txt";
            CheckPath(logpath);
            std::ofstream logfile(logpath, std::ios::app);

            for (auto t : join_storage.join_range)
            {
                int suffix = t.first;
                std::string savepath = saveprefix + "join_" + std::to_string(suffix) + ".bin";
                if (std::filesystem::exists(savepath))
                {
                    std::cout << "Groundtruths for " + savepath + " already exist! Skip." << std::endl;
                    continue;
                }
                CheckPath(savepath);

                std::ofstream outfile(savepath, std::ios::out | std::ios::binary);
                if (!outfile.is_open())
                    throw Exception("cannot open " + savepath);

                std::cout << "generating " << join_type << " join for ranges " << suffix << std::endl;
                logfile << join_type << " join, range " << suffix << std::endl;

                metric_gt_distances.store(0);
                timeval t1, t2;
                float searchtime = 0;
                for (int i = 0; i < join_storage.query_nb; i++)
                {
                    std::pair<int, int> r1 = t.second[i].first;
                    std::pair<int, int> r2 = t.second[i].second;

                    tbb::concurrent_vector<std::pair<int, int>> ans_list;
                    gettimeofday(&t1, NULL);
                    if (join_type == "mknn")
                        ans_list = ComputeMkNNJoinParallel(r1, r2, join_storage);
                    else if (join_type == "range")
                        ans_list = ComputeRangeJoinParallel(r1, r2, join_storage, dis_threshold);
                    else
                        throw Exception("Wrong join type: " + join_type);
                    gettimeofday(&t2, NULL);
                    auto duration = GetTime(t1, t2);
                    searchtime += duration;

                    int ans_nb = ans_list.size();
                    outfile.write((char *)&ans_nb, sizeof(int));
                    for (auto ans : ans_list)
                    {
                        outfile.write((char *)&ans, sizeof(std::pair<int, int>));
                    }
                }
                outfile.close();
                long long metric_gt_distances_nb = metric_gt_distances.load() / join_storage.query_nb;
                logfile << "Latency (per query): " << searchtime / join_storage.query_nb << " s, DCO (per query): " << metric_gt_distances_nb << std::endl;
            }

            logfile.close();
        }

        std::vector<std::pair<int, int>> ComputeMkNNJoin(std::pair<int, int> r1, std::pair<int, int> r2, JoinDataLoader &join_storage)
        {
            // Stage 1: find KNN of table2 for elements in table1
            std::vector<std::unordered_set<int>> table1_to_table2(join_storage.storage1->data_nb);
            for (int j = r1.first; j <= r1.second; j++)
            {
                std::priority_queue<std::pair<float, int>> knn;
                for (int k = r2.first; k <= r2.second; k++)
                {
                    float dis = dis_compute(join_storage.storage1->data_points[j], join_storage.storage2->data_points[k]);
                    knn.emplace(dis, k);
                    if (knn.size() > join_storage.query_K)
                        knn.pop();
                }

                while (!knn.empty())
                {
                    table1_to_table2[j].insert(knn.top().second);
                    knn.pop();
                }
            }

            // Stage 2: find KNN of table1 for elements in table2
            std::vector<std::unordered_set<int>> table2_to_table1(join_storage.storage2->data_nb);
            for (int j = r2.first; j <= r2.second; j++)
            {
                std::priority_queue<std::pair<float, int>> knn;
                for (int k = r1.first; k <= r1.second; k++)
                {
                    float dis = dis_compute(join_storage.storage1->data_points[k], join_storage.storage2->data_points[j]);
                    knn.emplace(dis, k);
                    if (knn.size() > join_storage.query_K)
                        knn.pop();
                }

                while (!knn.empty())
                {
                    table2_to_table1[j].insert(knn.top().second);
                    knn.pop();
                }
            }

            // Stage 3：find MKNN pairs
            std::vector<std::pair<int, int>> ans_list;
            for (int j = r1.first; j <= r1.second; j++)
            {
                for (int k : table1_to_table2[j])
                {
                    if (table2_to_table1[k].count(j))
                    {
                        ans_list.emplace_back(j, k);
                    }
                }
            }

            return ans_list;
        }

        tbb::concurrent_vector<std::pair<int, int>> ComputeMkNNJoinParallel(std::pair<int, int> r1, std::pair<int, int> r2, JoinDataLoader &join_storage)
        {
            using ConcurrentSet = tbb::concurrent_unordered_set<int>;
            using ConcurrentVector = tbb::concurrent_vector<std::pair<int, int>>;

            // Stage 1: 并行查找table1到table2的KNN
            std::vector<ConcurrentSet> table1_to_table2(join_storage.storage1->data_nb);
            tbb::parallel_for(r1.first, r1.second + 1, [&](int j)
                              {
                            long long local_dist=0;
                            std::priority_queue<std::pair<float, int>> knn;
                            for (int k = r2.first; k <= r2.second; k++) {
                                float dis = dis_compute(join_storage.storage1->data_points[j], 
                                                    join_storage.storage2->data_points[k]);
                                local_dist++;
                                knn.emplace(dis, k);
                                if (knn.size() > join_storage.query_K)
                                    knn.pop();
                            }
    
                            while (!knn.empty()) {
                                table1_to_table2[j].insert(knn.top().second);
                                knn.pop();
                            }
                        
                            metric_gt_distances.fetch_add(local_dist, std::memory_order_relaxed); });

            // Stage 2: 并行查找table2到table1的KNN
            std::vector<ConcurrentSet> table2_to_table1(join_storage.storage2->data_nb);
            tbb::parallel_for(r2.first, r2.second + 1, [&](int j)
                              {
                                long long local_dist=0;
                            std::priority_queue<std::pair<float, int>> knn;
                            for (int k = r1.first; k <= r1.second; k++) {
                                float dis = dis_compute(join_storage.storage1->data_points[k], 
                                                    join_storage.storage2->data_points[j]);
                                knn.emplace(dis, k);
                                if (knn.size() > join_storage.query_K)
                                    knn.pop();
                            }
                            
                            while (!knn.empty()) {
                                table2_to_table1[j].insert(knn.top().second);
                                knn.pop();
                            }
                             
                            metric_gt_distances.fetch_add(local_dist, std::memory_order_relaxed); });

            // Stage 3: 并行查找MKNN对
            ConcurrentVector ans_list;
            tbb::parallel_for(r1.first, r1.second + 1, [&](int j)
                              {
                            for (int k : table1_to_table2[j]) {
                                if (table2_to_table1[k].count(j)) {
                                    ans_list.push_back(std::make_pair(j, k));
                                }
                        } });

            return ans_list;
        }

        tbb::concurrent_vector<std::pair<int, int>> ComputeRangeJoinParallel(
            std::pair<int, int> r1,
            std::pair<int, int> r2,
            JoinDataLoader &join_storage,
            float dis_threshold)
        {
            using ConcurrentVector = tbb::concurrent_vector<std::pair<int, int>>;
            using BlockedRange = tbb::blocked_range<int>;
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 64);

            // 获取距离计算函数
            auto dist_func = space->get_dist_func();
            auto dist_func_param = space->get_dist_func_param();

            // 结果容器
            ConcurrentVector ans_list;

            tbb::parallel_for(
                tbb::blocked_range2d<int>(r1.first, r1.second + 1, 256,
                                        r2.first, r2.second + 1, 256),
                [&](const tbb::blocked_range2d<int>& r) {
                    for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                        for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                            float dis = dis_compute(join_storage.storage1->data_points[i],
                                                    join_storage.storage2->data_points[j]);
                            metric_gt_distances.fetch_add(1, std::memory_order_relaxed);
                            if (dis <= dis_threshold) {
                                ans_list.emplace_back(i, j);
                            }
                        }
                    }
                }
            );

            return ans_list;
        }
    };

    class TreeNode
    {
    public:
        int node_id;
        int lbound, rbound;
        int depth;
        std::vector<TreeNode *> childs;
        TreeNode(int l, int r, int d) : lbound(l), rbound(r), depth(d) {}
    };

    class SegmentTree
    {
    public:
        int ways_ = 2;
        TreeNode *root{nullptr};
        int max_depth{-1};
        std::vector<TreeNode *> treenodes;

        SegmentTree(int data_nb)
        {
            root = new TreeNode(0, data_nb - 1, 0);
        }

        void BuildTree(TreeNode *u)
        {
            if (u == nullptr)
                throw Exception("Tree node is a nullptr");
            treenodes.emplace_back(u);
            max_depth = std::max(max_depth, u->depth);
            int L = u->lbound, R = u->rbound;
            size_t Len = R - L + 1;
            if (L == R)
                return;
            int gap = (R - L + 1) / ways_;
            int res = (R - L + 1) % ways_;

            if (!gap)
            {
                for (int i = L; i <= R; ++i)
                {
                    TreeNode *child = new TreeNode(i, i, u->depth + 1);
                    u->childs.emplace_back(child);
                    BuildTree(child);
                }
            }
            else
            {
                for (int l = L; l <= R;)
                {
                    int r = l + gap - 1;
                    if (res > 0)
                    {
                        r++;
                        res--;
                    }
                    r = std::min(r, R);
                    TreeNode *childnode = new TreeNode(l, r, u->depth + 1);
                    u->childs.emplace_back(childnode);
                    BuildTree(childnode);
                    l = r + 1;
                }
            }
        }

        std::vector<TreeNode *> range_filter(TreeNode *u, int ql, int qr)
        {
            if (u->lbound >= ql && u->rbound <= qr)
                return {u};
            std::vector<TreeNode *> res;
            if (u->lbound > qr)
                return res;
            if (u->rbound < ql)
                return res;
            for (auto child : u->childs)
            {
                auto t = range_filter(child, ql, qr);
                while (t.size())
                {
                    res.emplace_back(t.back());
                    t.pop_back();
                }
            }
            return res;
        }

        std::vector<TreeNode *> filter_and_estimation(size_t &car, TreeNode *u, int ql, int qr)
        {
            if (u->lbound >= ql && u->rbound <= qr)
            {
                car += u->rbound - u->lbound + 1;
                return {u};
            }
            std::vector<TreeNode *> res;
            if (u->lbound > qr || u->rbound < ql)
                return res;
            for (size_t i = 0; i < u->childs.size(); ++i)
            {
                auto &child = u->childs[i];
                auto t = filter_and_estimation(car, child, ql, qr);
                res.insert(res.end(), t.begin(), t.end());
            }
            return res;
        }
    };
}