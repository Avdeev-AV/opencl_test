#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
//#define SIZE 32

const char * Kernelsource =                                                        "\n" \
"__kernel void sq(__global int* input, __global int* output, const unsigned int count)                    \n" \
"{                                                                                  \n" \
"    int i = get_global_id(0);                                                      \n" \
"    int block = get_group_id(0);                                                   \n" \
"    int thread = get_local_id(0);                                                  \n" \
"    printf(\"i'm from %d block %d thread (global index: %d)\\n\",block,thread,i);  \n" \
"    if (i < count)                                                                 \n" \
"       input[i] = input[i] + i;                                                     \n" \
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
        platform = platforms[0];
        delete[] platforms;
    }
    //Device
    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    cl_device_id device = NULL;
    
    if (0 < num_devices)
    {
        cl_device_id* devices = new cl_device_id[num_devices];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);
        device = devices[0];
        delete[] devices;
    }
    //Context
    cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    cl_context context = clCreateContextFromType((NULL == platform) ? NULL: properties, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

    size_t size = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
            

    //Command Queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, NULL);

    //Program
    size_t srclen[] = { strlen(Kernelsource) };
    cl_program program = clCreateProgramWithSource(context, 1, &Kernelsource, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "sq", NULL);
    //cl_kernel kernel = getKernel(Kernelsource, context, device_id);

    const size_t arr_size = 32;
    float data[arr_size];
    float results[arr_size];
    
    for (int i = 0; i < arr_size; i++)
    {
        data[i] = i;
        std::cout << data[i] << std::endl;
    }       

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * arr_size, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * arr_size, NULL, NULL);

    clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(float) * arr_size, data, 0, NULL, NULL);

    unsigned int count = arr_size;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &count);

    size_t group;

    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &arr_size, &group, 0, NULL, NULL);

    clFinish(command_queue);

    //int result[arr_size];
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
    for (int i = 0; i < arr_size; i++)
    {
        std::cout << results[i] << std::endl;
    }
    

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;
}