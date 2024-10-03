#include <iostream>
#include <memory>
#include <vector>


// bank_size: 32 一个bank只能是32位，或4个字节 
// 在一些能力强的CUDA上，bank_size是8个字节，64位
// num_banks: 32
// N:
void bank_id_1d_mapping(int bank_size, int num_banks, int N)
{
    for (int i = 0; i < N; i++)
    {
        int bank_idx = (i * sizeof(float) * 8 / bank_size) % num_banks;
        std::cout << "Array Idx: " << i << " "
                  << "Bank Idx: " << bank_idx << std::endl;
    }
}

int main()
{

    constexpr const int bank_size{32}; // bits
    constexpr const int num_banks{32};

    const int M{4};
    const int N{64};

    std::cout << "Bank ID Mapping 1D: N = " << N << std::endl;
    bank_id_1d_mapping(bank_size, num_banks, N);
    return 0;
}