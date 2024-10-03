#include <iostream>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 3

__global__ 
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixValB += in[(curRow * w + curCol) * 3];
                    pixValG += in[(curRow * w + curCol) * 3 + 1];
                    pixValR += in[(curRow * w + curCol) * 3 + 2];
                    pixels++;
                }
            }
        }
        out[(Row * w + Col) * 3] = (unsigned char)(pixValB / pixels);
        out[(Row * w + Col) * 3 + 1] = (unsigned char)(pixValG / pixels);
        out[(Row * w + Col) * 3 + 2] = (unsigned char)(pixValR / pixels);
    }
}

// CUDA内核函数，用于对图像进行模糊处理
__global__ void blurKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int pixelIndex = idy * width + idx;

    if (idx < width && idy < height) {
        for (int c = 0; c < channels; ++c) {
            int sum = 0;
            int count = 0;

            // 遍历周围的像素点，计算模糊
            for (int blur_x = -BLUR_SIZE; blur_x <= BLUR_SIZE; ++blur_x) {
                for (int blur_y = -BLUR_SIZE; blur_y <= BLUR_SIZE; ++blur_y) {
                    int curX = idx + blur_x;
                    int curY = idy + blur_y;
                    if (curX >= 0 && curX < width && curY >= 0 && curY < height) {
                        int curIndex = (curY * width + curX) * channels + c;
                        sum += d_in[curIndex];
                        count++;
                    }
                }
            }
            d_out[pixelIndex * channels + c] = sum / count;  // 取均值
        }
    }
}


int main() {
    cv::Mat inputImage = cv::imread("/home/ghr/code/Learn_CUDA/data/Lenna.jpg");

    if (inputImage.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat outputImage(height, width, CV_8UC3);

    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_input, inputImage.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 定义线程块和网格的大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    blurKernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, width, height,3);



    cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imshow("Input Image", inputImage);
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    return 0;
}