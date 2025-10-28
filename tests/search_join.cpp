#include "iRG_search.h"
#include "search_join.h"


std::unordered_map<std::string, std::string> paths;
std::unordered_map<std::string, float> radius_dict = {
    {"siftsmall", 120000},
    {"sift", 60012},
    {"gist", 0.596422},
    {"msong", 210.42},
    {"paper", 0.247342},
    {"WIT", 73.7635},
}; // e=0.1%

int main(int argc, char **argv)
{
    const int query_K = 10;
    int query_nb = 10;
    int query_modes = 10;
    int M;
    float lambda_threshold = 5.0;
    float dis_threshold = 0;
    int max_threads = 64;

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path1")
            paths["data_vector1"] = argv[i + 1];
        if (arg == "--data_path2")
            paths["data_vector2"] = argv[i + 1];
        if (arg == "--join_type")
            paths["join_type"] = argv[i + 1];
        if (arg == "--method_name")
            paths["method_name"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file1")
            paths["index1"] = argv[i + 1];
        if (arg == "--index_file2")
            paths["index2"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            max_threads = std::stoi(argv[i + 1]);
    }

    if (argc < 21)
        throw Exception("please check input parameters");

    VectorJoin::DataLoader storage1, storage2;
    storage1.LoadData(paths["data_vector1"]);
    storage2.LoadData(paths["data_vector2"]);

    std::filesystem::path data_path1(paths["data_vector1"]);
    std::filesystem::path data_path2(paths["data_vector2"]);
    std::string dataname1 = data_path1.stem().string();
    std::string dataname2 = data_path2.stem().string();

    if (dataname1.find("siftsmall") != std::string::npos)
    {
        dis_threshold = radius_dict["siftsmall"];
    }
    else if (dataname1.find("sift") != std::string::npos || dataname1.find("bigann_learn") != std::string::npos)
    {
        dis_threshold = radius_dict["sift"];
    }
    else if (dataname1.find("gist") != std::string::npos)
    {
        dis_threshold = radius_dict["gist"];
    }
    else if (dataname1.find("msong") != std::string::npos)
    {
        dis_threshold = radius_dict["msong"];
    }
    else if (dataname1.find("paper") != std::string::npos)
    {
        dis_threshold = radius_dict["paper"];
    }
    else if (dataname1.find("WIT") != std::string::npos)
    {
        dis_threshold = radius_dict["WIT"];
    }
    else
    {
        throw Exception("Unknown dataset: " + dataname1);
    }

    VectorJoin::JoinDataLoader join_storage(&storage1, &storage2, query_nb, query_modes);
    VectorJoin::JoinGenerator generator(join_storage.storage1->data_nb, join_storage.storage2->data_nb, join_storage.query_nb, join_storage.query_modes);

    join_storage.query_K = query_K;
    generator.GenerateJoinRange(paths["range_saveprefix"]);
    join_storage.LoadJoinRange(paths["range_saveprefix"]);

    std::cout << "Join Range Load Done!" << std::endl;
    generator.GenerateJoinGroundtruth(paths["groundtruth_saveprefix"], join_storage, paths["join_type"], dis_threshold);
    join_storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

    VectorJoin::iRangeGraph_Search<float> index1(paths["data_vector1"], paths["index1"], join_storage.storage1, M);
    VectorJoin::iRangeGraph_Search<float> index2(paths["data_vector2"], paths["index2"], join_storage.storage2, M);

    // searchefs can be adjusted
    std::vector<int> SearchEF = {1000, 750, 500, 300, 150, 100, 75, 50, 45, 40, 30, 20, 10};
    // std::vector<int> SearchEF = {500};
    std::vector<int> SearchEF_simjoin = {20, 10};

    if (paths["method_name"] == "ours")
    {
        std::string log_path = "./debug/log/log_" + paths["join_type"] + "_" + dataname1 + "_" + dataname2 + "_ours.txt";
        VectorJoin::ParaJoin<float> join_processor(&index1, &index2, M, &join_storage, paths["join_type"], log_path, lambda_threshold, dis_threshold, max_threads);
        join_processor.search(SearchEF, paths["result_saveprefix"], paths["join_type"], query_K, dis_threshold);
    }
    else
    {
        throw Exception("Unknown method name: " + paths["method_name"]);
    }
}