This project builds a C# console app which serves as a code generator for a few pieces of Whisper.dll and WhisperNet.dll.

Specifically, it generates two things.

1. It compresses the compiled DXBC shaders into a blob of bytes, and prints std::array with these bytes into shaderData-Release.inl and shaderData-Debug.inl C++ files.

2. It parses the `languageCodez.tsv`, and generates both C++ and C# code with the data from that table.

The tool uses relative paths across source files.
These paths will break if you move the source of the tool, or the source data of the tool.