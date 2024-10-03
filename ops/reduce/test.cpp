#include <iostream>
int main(){
    int xy = 9;
    std::cout << "xy>>1: " << (xy>>1) << std::endl; 
    std::cout << "xy<<1: " << (xy<<1) << std::endl; 

    xy<<=1;
    std::cout << "xy<<=1: " << xy << std::endl; 
    return 0;
}