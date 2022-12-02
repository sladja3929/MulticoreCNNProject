//globalsize : { ic, oc*i*j }
//localsize : { ic, 1 }
__kernel void convolution(__global float *inputs, __global float *filters, __local float *filterout, __global float *outputs, __global float *biases,
	int inputDim, int outputDim, int N) {
	int gi = get_global_id(1);
	int oc = gi / (N * N);	// output channel
	int ic = get_local_id(0);	// input channel && local id
	int j = gi % N;
	int i = (gi - oc * (N * N) - j) / N;	// global size {[oc][i][j],[ic]}를  설정함. 따라서 oc,i,j를 계산
	
	float sum = 0;
	//float *input = inputs + N * N * ic;
	//float *filter = filters + 3 * 3 * (oc * inputDim + ic);
	//float *output = outputs + N * N * oc;

	for (int k = 0; k < 3; k++) {
		for (int l = 0; l < 3; l++) {
			int x = i + k - 1;
			int y = j + l - 1;
			if (x >= 0 && x < N && y >= 0 && y < N)
				sum += inputs[N * N * ic + x * N + y] * filters[3 * 3 * (oc * inputDim + ic) + k * 3 + l];	// filter 계산한 후 local memory에 저장	
		}
	}

	filterout[ic] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	int l_size = get_local_size(0);
	float bias = biases[oc];

	float bias = biases[oc];
   if (ic == 0) {
      if (inputDim == 3)
         filterout[ic] += filterout[2];
      output[i * N + j] = (filterout[0] + bias) > 0 ? (filterout[0] + bias) : 0;
   }

	if (l_size == 3) {
		if (ic != 0) return;
		filterout[0] += filterout[1];
		filterout[0] += filterout[2];
		outputs[N * N * oc + i * N + j] = (filterout[0] + bias) > 0 ? (filterout[0] + bias) : 0;
	}
	else {
		for (int p = l_size / 2; p >= 1; p = p >> 1) {
			if (ic < p) filterout[ic] += filterout[ic + p];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (ic == 0) outputs[N * N * oc + i * N + j] = (filterout[0] + bias) > 0 ? (filterout[0] + bias) : 0;
	}
}

// global[d2][d1][n][n]
//float ReLU(float x) { return (x > 0) ? x : 0; }
//__kernel void convolution(__global float *inputs, __global float *outputs, __global float *filters, __global float *biases,
//__local float *l_mem, __private int d1, __private int d2, __private int n) {
//	int g_id = get_global_id(0);
//	int l_id = get_local_id(0);
//	int g_i = g_id / d2;	// d2
//	int g_j = g_id % d2 / d1;	// d1
//	int g_k = g_id % d1 / n; // n
//	int g_l = g_id % n;	// n
//
//	int i_off = n * n * g_j;
//	int o_off = n * n * g_i;
//	int f_off = 3 * 3 * (g_i * d1 + g_j);
//
//	float sum = 0;
//	for (int i = 0; i < 3; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			int x = g_k + i - 1;
//			int y = g_l + j - 1;
//			if (x >= 0 && x < n && y >= 0 && y < n) {
//				sum += inputs[i_off + x * n + y] * filters[f_off + i * 3 + j];
//			}
//		}
//	}
//
//	l_mem[l_id] = sum;
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	int l_s = get_local_size(0);
//	if (l_s == 3) {
//		if (l_id != 0) return;
//		l_mem[0] += l_mem[1];
//		l_mem[0] += l_mem[2];
//		outputs[o_off + g_k + g_l] = l_mem[0];
//	}
//	else {
//		for (int p = l_s >> 1; p >= 1; p >>= 1) {
//			if (l_id < p) l_mem[l_id] += l_mem[l_id + p];
//			barrier(CLK_LOCAL_MEM_FENCE);
//		}
//
//		if (l_id == 0) {
//			outputs[o_off + g_k + g_l] = l_mem[0];
//		}
//		else {
//			return;
//		}
//	}
//
//	outputs[o_off + g_k + g_l] = ReLU(outputs[g_k + g_l] + biases[g_i]);
//}
//