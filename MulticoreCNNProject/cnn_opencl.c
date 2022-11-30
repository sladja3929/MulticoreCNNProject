#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cnn.h"

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;

char* get_source_code(const char* file_name, size_t* len) {
    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t length = (size_t)ftell(file);
    rewind(file);

    char* source_code = (char*)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';
    fclose(file);
    *len = length;

    return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char* log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size + 1);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
}

void cnn_init() {
    // Platform ID
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    // Device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create Context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
}

#define ReLU(x) (((x)>0)?(x):0)
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
    // Create Command Queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_ERROR(err);

    // Create Program Object
    size_t kernel_source_size;
    char* kernel_source = get_source_code("fc_kernel.cl", &kernel_source_size);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    // Build Program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    build_error(program, device, err);
    CHECK_ERROR(err);

    //커널 오브젝트 생성
    cl_kernel kernel_fc = clCreateKernel(program, "fc", &err);
    CHECK_ERROR(err);

    //버퍼 오브젝트 생성
    cl_mem buf_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * M, NULL, &err);
    CHECK_ERROR(err);
    float tmp = 0;
    err = clEnqueueFillBuffer(context, buf_output, &tmp, sizeof(float), 0, sizeof(float) * M, 0, NULL, NULL);
    CHECK_ERROR(err);
    cl_mem buf_input = clCreateBuffer(context, CL_MEM_READ_ONLY || CL_MEM_COPY_HOST_PTR, sizeof(float) * N, input_neuron, &err);
    CHECK_ERROR(err);
    cl_mem buf_weights = clCreateBuffer(context, CL_MEM_READ_ONLY || CL_MEM_COPY_HOST_PTR, sizeof(float) * N * M, weights, &err);
    CHECK_ERROR(err);
    cl_mem buf_biases = clCreateBuffer(context, CL_MEM_READ_ONLY || CL_MEM_COPY_HOST_PTR, sizeof(float) * M, biases, &err);
    CHECK_ERROR(err);

    //커널 인자 넣기
    err = clSetKernelArg(kernel_fc, 0, sizeof(float) * M, buf_output);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 1, sizeof(float) * N, buf_input);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 2, sizeof(float) * N * M, buf_weights);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_fc, 3, sizeof(float)* M, buf_biases);
    CHECK_ERROR(err);

    size_t global_size[2] = { M, N };
    size_t local_size[2] = { M, N };
    clEnqueueNDRangeKernel(queue, kernel_fc, 2, NULL, global_size, local_size, 0, NULL, NULL);
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {
	/*
	 * TODO
	 * Implement here.
	 * Write classification results to labels and confidences.
	 * See "cnn_seq.c" if you don't know what to do.
	 */

}