//  __global__ 表示共享内存，并行调用的关键字，gpu核函数必须以此声明
//  n 代表conv kernel的个数，n=Cin*H*W
//  DType*  data_im 代表一个张量，为输入图像(Cin,H,W)，以_im为后缀的都与输入图像相关
//  feature map的高宽和kernel的高宽
template <typename DType>
__global__ void im2col_gpu_kernel(const int n, 
                                  const DType* data_im,                                                                                                             const int height, const int width,             //  feature map的高宽
                                  const int kernel_h, const int kernel_w,        //  kernel的高宽
                                  const int pad_h, const int pad_w,                  
                                  const int stride_h, const int stride_w,            
                                  const int dilation_h, const int dilation_w,        
                                  const int height_col, const int width_col,     //  输出column的高宽
                                  DType* data_col)                               //  输出column张量，(Kh * Kw* Cin, H, W)
{     
    for(int index = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){ 
        //  在0-gridDim.x*blockDim.x之间并行，到n结束，index代表第i个conv kernel的索引
        //  index index of output matrix
        //  说明一下，若%某个维度的大小，表示要索引这个维度上的位置，而/这个维度的大小则要索引下个维度的位置
        //  这里说的下个维度，是从右向左。如这里的data_im，shape为(Cin,H,W)，那么顺序为W，H，Cin
        //  是总的索引，%是在某个维度上某个范围内的索引
        const int h_index = index / width_col;
        const int h_col = h_index % height_col; // // 在某个c_in的维度下高的索引
        const int w_col = index % width_col;    // 在某个高的维度下宽的索引
        const int c_im = h_index / height_col;  // 输入通道索引
        const int c_col = c_im * kernel_h * kernel_w; // 输出通道索引
        const int h_offset = h_col * stride_h - pad_h; // 输出h的偏移
        const int w_offset = w_col * stride_w - pad_w; // 输出w的偏移
        DType* data_col_ptr = data_col;  //获得输出张量的指针拷贝
        // 指针向前移动，由于index是0-Cin*H*W，c_col,h_col和w_col有Cin、H和W种取值，正好对应index
        data_col_ptr += (c_col * height_col + h_col) * width_col + w_col; 
        const DType* data_im_ptr = data_im; //  获取输入张量的指针拷贝
        data_im_ptr += (c_im * height + h_offset) * width + w_offset; //指针向前移动
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) { // 对单个kernel进行循环
                int h_im = h_offset + i * dilation_h;
                int w_im = w_offset + j * dilation_w;
                // *+指针是只取指针所指位置的数值，这里赋值给对应位置
                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? // 若索引不越界
                                data_im_ptr[i * dilation_h * width + j * dilation_w] : 
                                static_cast<DType>(0);
                data_col_ptr += height_col * width_col;
            }
        }
}
