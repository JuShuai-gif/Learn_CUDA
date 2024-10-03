#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

__device__ int BLUR_SIZE = 1;

// CUDA内核函数，用于将图像转换为灰度图
__global__ void rgbToGrayKernel(unsigned char *d_in, unsigned char *d_out, int width, int height, int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = idy * width + idx;

    if (idx < width && idy < height)
    {
        int rgbIndex = pixelIndex * channels;
        unsigned char r = d_in[rgbIndex];
        unsigned char g = d_in[rgbIndex + 1];
        unsigned char b = d_in[rgbIndex + 2];
        d_out[pixelIndex] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// CUDA内核函数，用于对图像进行模糊处理
__global__ void blurKernel(unsigned char *d_in, unsigned char *d_out, int width, int height, int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = idy * width + idx;

    if (idx < width && idy < height)
    {
        for (int c = 0; c < channels; ++c)
        {
            int sum = 0;
            int count = 0;

            // 遍历周围的像素点，计算模糊
            for (int blur_x = -BLUR_SIZE; blur_x <= BLUR_SIZE; ++blur_x)
            {
                for (int blur_y = -BLUR_SIZE; blur_y <= BLUR_SIZE; ++blur_y)
                {
                    int curX = idx + blur_x;
                    int curY = idy + blur_y;
                    if (curX >= 0 && curX < width && curY >= 0 && curY < height)
                    {
                        int curIndex = (curY * width + curX) * channels + c;
                        sum += d_in[curIndex];
                        count++;
                    }
                }
            }
            d_out[pixelIndex * channels + c] = sum / count; // 取均值
        }
    }
}

void rgbToGray(const cv::Mat &input, cv::Mat &output)
{
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    printf("%d\n", channels);
    unsigned char *d_in;
    unsigned char *d_out;
    size_t imgSize = width * height * channels * sizeof(unsigned char);
    size_t imgGraySize = width * height * sizeof(unsigned char);

    // 分配CUDA内存
    cudaMalloc((void **)&d_in, imgSize);
    cudaMalloc((void **)&d_out, imgGraySize);

    // 将输入图像数据从主机复制到设备
    cudaMemcpy(d_in, input.ptr(), imgSize, cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用灰度内核
    // rgbToGrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, width, height, channels);

    // 调用模糊内核
    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, width, height, channels);

    // 等待内核完成
    cudaDeviceSynchronize();

    // 将处理后的灰度图像数据从设备复制到主机
    cudaMemcpy(output.ptr(), d_out, imgGraySize, cudaMemcpyDeviceToHost);

    // 释放CUDA内存
    cudaFree(d_in);
    cudaFree(d_out);
}

int main(int argc, char **argv)
{

    std::string img_path{"/home/ghr/code/Learn_CUDA/data/Lenna.jpg"};

    // 加载图像
    cv::Mat input = cv::imread(img_path, cv::IMREAD_COLOR);
    if (input.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // 创建输出灰度图像
    cv::Mat output(input.rows, input.cols, CV_8UC1);

    // 调用CUDA灰度化函数
    rgbToGray(input, output);

    // 显示原始图像和灰度图像
    cv::imshow("Original Image", input);
    cv::imshow("Grayscale Image", output);
    cv::waitKey(0);

    return 0;
}
