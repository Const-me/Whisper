// TODO: compile another version of these shader, and use it on GPUs with ExtendedDoublesShaderInstructions flag, will become slightly faster
// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_feature_data_d3d11_options
#ifndef ExtendedDoublesShaderInstructions
#define ExtendedDoublesShaderInstructions 0
#endif

// Compute num/den in FP64 precision
inline double div64( double num, double den )
{
#if ExtendedDoublesShaderInstructions
	return num / den;
#else
	// https://en.wikipedia.org/wiki/Division_algorithm#Newton%E2%80%93Raphson_division
	double x = 1.0f / (float)den;
	x += x * ( 1.0 - den * x );
	x += x * ( 1.0 - den * x );
	return num * x;
#endif
}

// Compute sqrt(x) in FP64 precision
inline double sqrt64( double x )
{
	double root = sqrt( (float)x );
	root = 0.5 * ( root + div64( x, root ) );
	root = 0.5 * ( root + div64( x, root ) );
	return root;
}