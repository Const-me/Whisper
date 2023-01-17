This folder contains 2 sample speech audio clips, `jfk.wav` and `columbia.wma`

jfk.wav is from whisper.cpp repository.

columbia.wma is from Wikipedia: https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
I re-encoded the audio from Ogg Vorbis into Windows Media Audio, because Media Foundation is unable to decode Vorbis.

The rest of the text files in this folder are the outputs of the in-app performance profiler, when the app was transcribing these two audio clips on three different computers.

The “1080ti” files are from my desktop, which has nVidia GeForce 1080Ti GPU.

The “vega7” files are from my laptop, the GPU is integrated into AMD Ryzen 5 5600U processor.
The laptop model is HP ProBook 445 G8. While running the tests, the laptop was on battery power.

The “1650” files are from another desktop with nVidia GeForce 1650.

The file names with “medium” in the middle were made with “ggml-medium.bin” Whisper model, with “large” were made with “ggml-large.bin” model.

In theory, 1080ti delivers 10.6 TFlops FP32 and 484.4 GB/second VRAM bandwidth.

That variant of 1650 delivers 2.6 TFlops FP32, and 128.1 GB/second VRAM bandwidth.

The AMD APU in that laptop delivers 1.6 TFlops FP32, and 51.2 GB/second memory bandwidth.