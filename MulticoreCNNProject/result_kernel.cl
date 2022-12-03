// global: { P * N }
// local: { N }
__kernel void result(__global float *output, __local float *l_mem, __private int n) {
	int g_id = get_global_id(0);
	int l_id = get_local_id(0);
	int l_size = get_local_size(0);

	l_mem[l_id] = (g_id < n) ? output[g_id] : 0;	// pading
	barrier(CLK_LOCAL_MEM_FENCE);

	float max = 0;
	for (int p = l_size >> 1; p >= 1; p >>= 1) {
		if (l_id < p) l_mem[l_id] = (l_mem[l_id] > l_mem[l_id + p]) ? l_mem[l_id] : l_mem[l_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	l_sum[l_id] = (g_id < n) ? output[g_id] : 0;	// pading

	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}