This library implements high-performance GPGPU inference of OpenAI's Whisper automatic speech recognition (ASR) model.

The library requires a hardware GPU which supports Direct3D 11.0, a 64-bit Windows OS, only works within 64-bit processes, and requires a 64 bit CPU which supports [SSE 4.1](https://en.wikipedia.org/wiki/SSE4#SSE4.1) and [F16C](https://en.wikipedia.org/wiki/F16C) extensions.

The main entry point of the llibrary is `Whisper.Library` static class.
Call `loadModel` function from that class to load an ML model from a binary file.

These binary files are available for free download on [Hugging Face]( https://huggingface.co/datasets/ggerganov/whisper.cpp).
I recommend `ggml-medium.bin` (1.42GB in size), because I’ve mostly tested the software with that model.

Once the model is loaded, create a context by calling `createContext` extension method,
then use that object to transcribe or translate multimedia files or realtime audio captured from microphones.