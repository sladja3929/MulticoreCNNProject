__kernel void pooling(__global float *inputs, __global float *outputs, __private int n) {
	int m = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);

	float max = 0, pixel;
	int i_i = m * (n * n * 4);
	for (int k = 0; k < 2; ++k) {
		int i_j = (i * 2 + k) * (2 * n);
		for (int l = 0; l < 2; ++l) {
			pixel = inputs[i_i + i_j + (j * 2 + l)];
			max = (max > pixel) ? max : pixel;
		}
	}
	outputs[(m * n * n) + (i * n) + j] = max;
}