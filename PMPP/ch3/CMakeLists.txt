
# 寻找OpenCV包
find_package(OpenCV REQUIRED)

add_executable(PMPP_ch3_3-9-8 3-9-8.cpp)

# 添加可执行文件
add_executable(PMPP_ch3_color2grey color2grey.cu)

# 链接OpenCV库
target_link_libraries(PMPP_ch3_color2grey ${OpenCV_LIBS})

# 添加可执行文件
add_executable(PMPP_ch3_blurKernel blurKernel.cu)

# 链接OpenCV库
target_link_libraries(PMPP_ch3_blurKernel ${OpenCV_LIBS})