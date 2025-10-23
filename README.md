# ParaJoin

This repo is the implementation of ParaJoin: A Parallelism-aware Approximate Filtered Similarity Join Method in Vector Databases

## Requirements

- A modern C++ compiler that supports C++17
- CMake 3.23.1 or higher
- OpenMP, TBB, Faiss

## Quick Start

### Build

```bash
mkdir build && cd build && cmake .. && make
```

### Construct Index

#### parameters:

**`--data_path`**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(float)` bytes contain the contents of the data one data point in a time. 
The data points should be already sorted in ascending order by the attribute.

**`--index_file`**: The constructed index will be saved to this file, in .bin format.

**`--M`**: The degree of the graph index.

**`--ef_construction`**: The size of result set during index building.

**`--threads`**: The number of threads for index building.


#### command:
```bash
./build/tests/buildindex \
--data_path [path to data points] \
--index_file [file path to save index] \
--M [integer] \
--ef_construction [integer] \
--threads [integer]
```

### Running example

#### parameters:

**`--data_path1`**: The .bin format data path for dataset A

**`--data_path2`**: The .bin format data path for dataset B

**`--method_name`**: join algorithm, 'ours' for default

**`--join_type`**: 'mknn' or 'range'

**`--range_saveprefix`**: The path of folder where query range files will be saved. 0~9 denotes query range fractions, respectively.

**`--groundtruth_saveprefix`**: The path of folder where groundtruth files will be saved.

**`--index_file1`**: The index file path for dataset A, in .bin format. 

**`--index_file2`**: The index file path for dataset B, in .bin format. 

**`--result_saveprefix`**: The path of folder where result files will be saved.

**`--M`**: The degree of the graph index. It should equal the 'M' used for constructing index.

**`--threads`**: The number of threads for parallel join.

#### command:
```bash
./build/tests/search_join \
--data_path1 [path to dataset A] \
--data_path2 [path to dataset B] \
--method_name ours \
--join_type [mknn or range] \
--range_saveprefix [folder path to save join ranges]  \
--groundtruth_saveprefix [folder path to save groundtruth] \
--index_file1 [path of the index file A] \
--index_file2 [path of the index file B] \
--result_saveprefix [folder path to save results] \
--M [integer] \
--threads [integer]
```

## Datasets
| Dataset |Object num| Dimension | Type |
|---------|-----------|-----------|----------------|
|[SIFT](http://corpus-texmex.irisa.fr/)|   1,000,000    |    128    | Image + Attribute |
|[GIST](http://corpus-texmex.irisa.fr/)|   1,000,000    |    960    | Image + Attribute |
|[Msong](https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS?tab=readme-ov-file)|   992,272   |   420     |Audio + Attribute|
|[Paper](https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS?tab=readme-ov-file)|   2,029,997  |   200     |   Text+Attribute    |
|[WIT](https://github.com/google-research-datasets/wit)   |   1,000,000   |   2048    |   Image + Attribute   |
|[SIFT100M](http://corpus-texmex.irisa.fr/)|   100,000,000    |    128    | Image + Attribute |