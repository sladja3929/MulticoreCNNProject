// global: { d, n, n }
__kernel void pooling(__global float *inputs, __global float *outputs, __private int n) {
	int g_id = get_global_id(0);
	int g_i = g_id / (n * n);
	int g_j = g_id / n % n;
	int g_k = g_id % n;

	float *input = inputs + g_i * n * n * 4;
	float *output = outputs + g_i * n * n;

	float max = 0, pixel;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			pixel = input[((g_j * 2 + i) * (2 * n)) + (g_k * 2 + j)];
			max = (max > pixel) ? max : pixel;
		}
	}
	output[g_j * n + g_k] = max;
}