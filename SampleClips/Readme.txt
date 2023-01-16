This folder contains 2 sample speech audio clips, `jfk.wav` and `columbia.wma`

jfk.wav is from whisper.cpp repository.

columbia.wma is from Wikipedia: https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
I had to re-encoded the audio from Ogg Vorbis into Windows Media Audio, because Media Foundation is unable to decode Vorbis.

The rest of the text files in this folder are the outputs of the in-app performance profiler, when the app was transcribing these two audio clips on two different computers.

The files names containing `1080ti` are from my desktop, which has nVidia GeForce 1080Ti GPU.

The files names containing `vega7` are from my laptop, the GPU is integrated into AMD Ryzen 5 5600U processor.
The laptop model is HP ProBook 445 G8. While running the tests, the laptop was on battery power.

The files names with `medium` in the middle were made with `ggml-medium.bin` Whisper model.

The files with `large` were made with `ggml-large.bin` model.