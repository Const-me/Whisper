This C++ project builds a DLL which actually does the heavy lifting of this project.

It implements the ML model, handles multimedia files with Media Foundation, captures audio (also with MF), does voice activity detection (custom code running on CPU), and a few smaller things.

The code requires C++/20, and only tested with Visual Studio 2022.

When running pure GPGPU model, the DLL requires SSE 4.1 instruction set.

When running a hybrid model, the DLL requires AVX1, FMA3, F16C, and BMI1 instruction set extensions.