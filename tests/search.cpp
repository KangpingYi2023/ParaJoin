#include "iRG_search.h"

std::unordered_map<std::string, std::string> paths;

const int query_K = 10;
int M;

void Generate(VectorJoin::DataLoader &storage)
{
    storage.LoadData(paths["data_vector"]);
    VectorJoin::QueryGenerator generator(storage.data_nb, storage.query_nb);
    generator.GenerateRange(paths["range_saveprefix"]);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    generator.GenerateGroundtruth(paths["groundtruth_saveprefix"], storage);
}

void init()
{
    // data vectors should be sorted by the attribute values in ascending order
    paths["data_vector"] = "";

    paths["query_vector"] = "";
    // the path of document where range files are saved
    paths["range_saveprefix"] = "";
    // the path of document where groundtruth files are saved
    paths["groundtruth_saveprefix"] = "";
    // the path where index file is saved
    paths["index"] = "";
    // the path of document where search result files are saved
    paths["result_saveprefix"] = "";
    // M is the maximum out-degree same as index build
}

int main(int argc, char **argv)
{
    // init();

    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--range_saveprefix")
            paths["range_saveprefix"] = argv[i + 1];
        if (arg == "--groundtruth_saveprefix")
            paths["groundtruth_saveprefix"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--result_saveprefix")
            paths["result_saveprefix"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
    }

    if (argc != 15)
        throw Exception("please check input parameters");

    VectorJoin::DataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);
    // If it is the first run, Generate shall be called; otherwise, Generate can be skipped
    Generate(storage);
    storage.LoadQueryRange(paths["range_saveprefix"]);
    storage.LoadGroundtruth(paths["groundtruth_saveprefix"]);

    VectorJoin::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);
    // searchefs can be adjusted
    std::vector<int> SearchEF = {1700, 1000, 700, 400, 300, 200, 150, 100, 50, 40, 30, 20, 10};
    index.search(SearchEF, paths["result_saveprefix"], M);
}