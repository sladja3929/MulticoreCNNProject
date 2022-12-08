// globalsize: { P * M, N }
// localsize: { 1, N }

__kernel void fc(__global float *output, __constant float *input, __constant float *weights, __constant float *biases, __local float *l_sum, __private int PARALLEL) {
	int g_id = get_global_id(0);
	int n = get_global_size(1);
	int m = get_global_size(0) / PARALLEL;
	int g_i = g_id / m;	// PARALLEL
	int g_j = g_id % m;
	int l_id = get_local_id(1);

	l_sum[l_id] = input[g_i * n + l_id] * weights[g_j * n + l_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = n >> 1; p >= 1; p >>= 1) {
		if (l_id < p) l_sum[l_id] += l_sum[l_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0) {
		int o_idx = g_i * m + g_j;
		float result = l_sum[0] + biases[g_j];
		result = (result > 0) ? result : 0;
		output[o_idx] = result;
	}
}
