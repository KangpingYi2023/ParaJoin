#pragma once

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include "memory.hpp"

namespace searcher
{
    template <typename Block = uint64_t>
    struct Bitset
    {
    private:
        constexpr static int block_size = sizeof(Block) * 8;
        int nbytes;
        Block *data;

    public:
        explicit Bitset(int n)
            : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
              data(static_cast<uint64_t *>(memory::align_mm<64>(nbytes)))
        {
            std::memset(data, 0, nbytes);
        }

        ~Bitset() { free(data); }

        void set(int i)
        {
            data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
        }

        bool get(int i)
        {
            return (data[i / block_size] >> (i & (block_size - 1))) & 1;
        }

        void *block_address(int i) { return data + i / block_size; }
    };

    template <typename Block = uint64_t>
    struct AtomicBitset
    {
    private:
        using AtomicBlock = std::atomic<Block>;
        static constexpr int block_size = sizeof(Block) * 8;
        static constexpr int cache_line_size = 64; // 现代CPU缓存行大小

        int num_blocks;
        AtomicBlock *data; // 原子块数组

    public:
        explicit AtomicBitset(int n)
            : num_blocks((n + block_size - 1) / block_size),
              data(static_cast<AtomicBlock *>(memory::align_mm<cache_line_size>(num_blocks * sizeof(AtomicBlock))))
        {
            for (int i = 0; i < num_blocks; ++i)
            {
                data[i].store(0, std::memory_order_relaxed);
            }
        }

        ~AtomicBitset()
        {
            if (data)
            {
                free(data);
                data = nullptr;
            }
        }

        AtomicBitset(const AtomicBitset &) = delete;
        AtomicBitset &operator=(const AtomicBitset &) = delete;

        void set(int i)
        {
            const int block_idx = i / block_size;
            const int bit_offset = i % block_size;
            AtomicBlock &block = data[block_idx];

            Block old_val, new_val;
            do
            {
                old_val = block.load(std::memory_order_relaxed);
                new_val = old_val | (Block(1) << bit_offset);
            } while (!block.compare_exchange_weak(old_val, new_val,
                                                  std::memory_order_release, std::memory_order_relaxed));
        }

        bool get(int i) const
        {
            const int block_idx = i / block_size;
            const int bit_offset = i % block_size;

            // 原子读取（保证内存可见性）
            Block val = data[block_idx].load(std::memory_order_acquire);
            return (val >> bit_offset) & 1;
        }

        void clear()
        {
            for (int i = 0; i < num_blocks; ++i)
            {
                data[i].store(0, std::memory_order_relaxed);
            }
        }

        void *block_address(int i)
        {
            return data + i / block_size;
        }
    };

    template <typename dist_t = float>
    struct Candidiate
    {
        int id;
        dist_t distance;

        Candidiate() = default;
        Candidiate(int id, dist_t distance) : id(id), distance(distance) {}

        inline friend bool operator<(const Candidiate &lhs, const Candidiate &rhs)
        {
            return lhs.distance < rhs.distance ||
                   (lhs.distance == rhs.distance && lhs.id < rhs.id);
        }
        inline friend bool operator>(const Candidiate &lhs, const Candidiate &rhs)
        {
            return !(lhs < rhs);
        }
    };

    struct LinearPool
    {
    public:
        int nb, size_ = 0, cur_ = 0, capacity_;
        std::vector<Candidiate<float>, memory::align_alloc<Candidiate<float>>> data_;
        Bitset<uint64_t> vis;
        constexpr static int kMask = 2147483647;

        LinearPool(int n, int capacity)
            : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

        bool insert(int u, float dist)
        {
            if (size_ == capacity_ && dist >= data_[size_ - 1].distance)
            {
                return false;
            }
            int lo = find_bsearch(dist);
            std::memmove(&data_[lo + 1], &data_[lo],
                         (size_ - lo) * sizeof(Candidiate<float>));
            data_[lo] = {u, dist};
            if (size_ < capacity_)
            {
                ++size_;
            }
            if (lo < cur_)
            {
                cur_ = lo;
            }
            return true;
        }

        int pop()
        {
            set_checked(data_[cur_].id);
            int pre = cur_;
            while (cur_ < size_ && is_checked(data_[cur_].id))
            {
                ++cur_;
            }
            return get_id(data_[pre].id);
        }

        bool has_next() const { return cur_ < size_; }

        int get_size() const { return size_; }

        int id(int i) const { return get_id(data_[i].id); }

    private:
        int find_bsearch(float dist)
        {
            int lo = 0, hi = size_;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (data_[mid].distance > dist)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid + 1;
                }
            }
            return lo;
        }

        int get_id(int id) const { return id & kMask; }
        void set_checked(int &id) { id |= 1 << 31; }
        int is_checked(int id) { return id >> 31 & 1; }
    };
}