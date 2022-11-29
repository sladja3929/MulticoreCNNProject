__kernel void pooling2x2(__global float *input, __global float *output, __private int n) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	float max = 0;
	int k = 0, l = 0;
	for (; k < 2; ++k) {
		for (; l < 2; ++l) {
			float pixel = input[(i * 2 + k) * 2 * n + j * 2 + l];
			max = (max > pixel) ? max : pixel;
		}
	}
	output[i * n + j] = max;
}

__kernel void pooling(__global float *inputs, __global float *outputs, __private int d, __private int n) {
	int m = get_global_id(0);
	int i = get_global_id(1);
	int j = get_global_id(2);

	float max = 0, pixel;
	int k = 0, l = 0;
	for (; k < 2; ++k) {
		for (; l < 2; ++l) {
			pixel = inputs[(m * n * n * 4) + (i * 2 + k) * 2 * n + j * 2 + l];
			max = (max > pixel) ? max : pixel;
		}
	}
	outputs[(m * n * n) + i * n + j] = max;
}
