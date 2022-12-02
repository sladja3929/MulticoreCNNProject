// global = { m, n }
// local = { 1, n/2 }
// g_sum = { m, 2 }
// output = { m, 1 }
__kernel void fc(__global float *output, __global float *input, __global float *weights, __local float *l_sum) {
	int i = get_global_id(1);
	int j = get_global_id(0);
	int N = get_global_size(1);
	int M = get_global_size(0);
	int l_id = get_local_id(1);

	l_sum[l_id] = input[i] * weights[j * N + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = N >> 2; p >= 1; p >>= 1) {
		if (l_id < p) l_sum[l_id] += l_sum[l_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (l_id == 0) {
		output[j] += l_sum[0];
		//atomic_add(&output[j], l_sum[0]);
		//int gr_j = get_group_id(1);
		//g_sum[j * 2 + gr_j] = l_sum[0];
		//barrier(CLK_GLOBAL_MEM_FENCE);
		//if (gr_j == 0) {
		//	g_sum[j * 2] += g_sum[j * 2 + 1];
		//	output[j] = g_sum[j * 2];
		//	output[j] += biases[j];
		//	output[j] = (output[j] > 0) ? output[j] : 0;
		//}
	}
}


//__kernel void fc(__global float* output, __global float* input, __global float* weights, __local float* sum) {
//	int N = get_global_size(1);
//	int j = get_global_id(0);
//	int i = get_global_id(1);
//	int k = get_local_id(1);
//
//	sum[k] = input[i] * weights[j * N + i];
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	for (int p = get_local_size(1) / 2; p >= 1; p = p >> 1) {
//		if (k < p) sum[k] += sum[k + p];
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//	
//	if (k == 0) {
//		output[j] += sum[k];
//	}
//}