__kernel void fc(__global float* output, __global float* input, __global float* weights) {
	int M = get_global_size(0);
	int N = get_global_size(1);
	int j = get_global_id(0);
	int i = get_global_id(1);

	output[i * M + j] = input[i] * weights[j * N + i];
}

__kernel void reduction (__global float* output) {
	int M = get_global_size(0);
	int j = get_global_id(0);
	int i = get_global_id(1);

	for (int p = get_global_size(1) / 2; p >= 1; p = p >> 1) {
		if (i < p) output[i * M + j] += output[(i + p) * M + j];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}