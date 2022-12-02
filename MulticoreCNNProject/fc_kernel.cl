__kernel void fc(__global float *output, __global float *input, __global float *weights, __global float *biases, __local float *l_sum) {
	int i = get_global_id(1);
	int j = get_global_id(0);
	int N = get_global_size(1);
	int M = get_global_size(0);
	int l_id = get_local_id(1);

	l_sum[l_id] = input[i] * weights[j * N + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = N >> 1; p >= 1; p >>= 1) {
		if (l_id < p) l_sum[l_id] += l_sum[l_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0) {
		output[j] = l_sum[0];
		output[j] += biases[j];
		output[j] = (output[j] > 0) ? output[j] : 0;
	}
}
