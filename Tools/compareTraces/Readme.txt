This project builds a C++ console tool which compares debug traces of the model.

Tracing files easily exceed 1GB, and by default they’re disabled with a preprocessor macro in stdafx.h of the Whisper project.

When enabled, the main GPU implementation saves a trace into C:\Temp\2remove\Whisper\gpu.bin

The reference CPU implementation saves a trace into C:\Temp\2remove\Whisper\ref.bin

This code in this project is optimized for development speed. For this reason it requires AVX2 CPU, uses memory-mapped IO instead of proper parsing, and checks little to no errors.