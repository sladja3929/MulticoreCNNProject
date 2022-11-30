__kerel void fc(__global float* output, __global float* input, __global float* weights, __global float* biases) {
	int M = get_global_size(0);
	int N = get_global_size(1);
	int i = get_global_id(0);
	int j = get_global_id(1);

	output[i] += input[j] * weights[i * N + j];
	output[i] += biases[i];
}