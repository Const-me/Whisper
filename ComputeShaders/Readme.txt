This project compiles all the compute shaders which implement the model.

Many shaders come in 2 versions, something.hlsl and something64.hlsl

The version with the `64` suffix is used on AMD GPUs, the version without suffix is used on nVidia and Intel GPUs.

Not all of these shaders are actually used for anything.
Some of them are implementing binary compatibility for the reference CPU version, and not used unless messing with the `constexpr` flags in MlContext C++ class.
Such shaders often require FP64 support, which is an optional feature in D3D11.
CompressShaders tool detects such shaders by looking at the SFI0 chunk in the binary, and outputs a bitmap of the FP64 shaders.
This way, missing FP64 hardware support shouldn’t break the library.