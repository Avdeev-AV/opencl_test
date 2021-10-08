#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#define SIZE 128

const char* source =
"__kernel                                   \n"
"void vecAdd(__global int *a)               \n"
"{                                          \n"
"    int idx = get_global_id(0);            \n"
"    a[idx] = a[idx] + idx;                 \n"
"}                                          \n";

int main(void) {

    cl_platform_id* platforms = NULL;
    cl_uint     num_platforms;

    clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id*)
        malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_device_id* device_list = NULL;
    cl_uint           num_devices;

    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    device_list = (cl_device_id*)
        malloc(sizeof(cl_device_id) * num_devices);
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);


    cl_context context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);

    size_t srclen[] = { strlen(source) };
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, NULL);

    clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "source", NULL);

    float data[SIZE];
    float result[SIZE];

    for (int i = 0; i < SIZE; i++)
        data[i] = rand();


    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * SIZE, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * SIZE, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, SIZE * sizeof(float), NULL, 0, NULL, NULL);
    

    for (cl_uint i = 0; i < num_platforms; ++i) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
            128, platformName, nullptr);
        std::cout << platformName << std::endl;
    }

    unsigned int count = SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);

    size_t group;
    size_t global_size = SIZE;
    clGetKernelWorkGroupInfo(kernel, device_list[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof (size_t), &group, NULL);
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &group, 0, NULL, NULL);
    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, count * sizeof(float), result, 0, NULL, NULL);

    for (int i = 0; i < SIZE; i++)
        std::cout << result[i] << std::endl;


    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}