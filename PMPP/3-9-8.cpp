#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

int main()
{

    std::vector<float> nums{2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9};
    float sum{0.0};
    float sss{0.0};
    float max_elem{3.0};
    for (size_t i = 0; i < nums.size(); ++i)
    {
        sss+=max_elem - nums[i];
        sum += nums[i];
    }
    float baifenbi = sss/sum;
    printf("%f\n",baifenbi);



    return 0;
}