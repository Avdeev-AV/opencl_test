#include <iostream>
#include <chrono>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

    //Kernel
const char * source =                                                                    "\n" \
"__kernel void code (__global int * input, __global int * output, const unsigned int count)          \n" \
"{                                                                                              \n" \
"    int i = get_global_id(0);                                                                  \n" \
"    int block = get_group_id(0);                                                               \n" \
"    int thread = get_local_id(0);                                                              \n" \
"    printf(\"i'm from %d block %d thread (global index: %d)\\n\",block,thread,i);              \n" \
"    if (i < count)                                                                             \n" \
"       output[i] = input[i] + i;                                                                \n" \
"}";

int main() {
    //Platform
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);

    cl_platform_id platform = NULL;

    if (0 < num_platforms)
    {
        cl_platform_id* platforms = new cl_platform_id[num_platforms];
        clGetPlatformIDs(num_platforms, platforms, NULL);
        platform = platforms[1];
        char platformName[128];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, NULL);
        std::cout << platformName << " " << num_platforms << std::endl;
        delete[] platforms;
    }
    //Device
    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id device = NULL;
    
    if (0 < num_devices)
    {
        cl_device_id* devices = new cl_device_id[num_devices];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr);
        device = devices[0];
        char deviceName[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, NULL);
        std::cout << deviceName << " " << num_devices << std::endl;
        delete[] devices;
    }
    //Context
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContextFromType((NULL == platform) ? NULL: properties, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, 0);
            

    //Command Queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, NULL);

    //Program
    size_t srclen[] = { strlen(source) };
    cl_program program = clCreateProgramWithSource(context, 1, &source, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    //Kernel
    cl_kernel kernel = clCreateKernel(program, "code", NULL);

    const size_t arr_size = 2048;
    int results[arr_size];

    //Input array
    int data[arr_size];
    for (int i = 0; i < arr_size; i++)
    {
        data[i] = i;
    }       

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * arr_size, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * arr_size, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(float) * arr_size, data, 0, NULL, NULL);

    unsigned int count = arr_size;

    //Kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    
    //Launch kernel
    size_t group = 128;
    //clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
    auto start = std::chrono::steady_clock::now();
    int err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &arr_size, &group, 0, NULL, NULL);
    if (err)
    {
        std::cout << "Something went wrong here " << err << std::endl;
    }
    clFinish(command_queue);
    auto end = std::chrono::steady_clock::now();
    auto result_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
    
    for (int i = 0; i < arr_size; i++)
    {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl << "time = " << result_time << std::endl;
    
    //Release
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;
}