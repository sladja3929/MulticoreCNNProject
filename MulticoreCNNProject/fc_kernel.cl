__kernel void fc(__global float* output, __global float* input, __global float* weights, __local float* sum) {
	int N = get_global_size(1);
	int j = get_global_id(0);
	int i = get_global_id(1);
	int k = get_local_id(1);

	sum[k] = input[i] * weights[j * N + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = get_local_size(1) / 2; p >= 1; p = p >> 1) {
		if (k < p) sum[k] += sum[k + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (k == 0) {
		output[j] += sum[k];
	}
}