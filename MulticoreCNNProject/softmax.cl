__kernel void softmax(__global float* output, __local float max, __local float sum) {
	int l_i = get_local_id(0);
	int g_i = get_global_id(0);
	int n = get_local_size(0);
	int i;

	if(l_i == 0) {
		max = output[g_i];
		sum = 0;

		for (i = 1; i < n; i++) {
			max = (output[g_i + i] > max) ? output[g_i + i] : max;
		}

		for(i = 0; i < n; i++) {
			sum += exp(output[g_i + i] - max);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	output[g_i] = exp(output[g_i] - max) / sum;
}