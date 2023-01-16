#pragma once

namespace ComLight
{
	class Exception : public std::runtime_error
	{
		// I don't like C++ exceptions too much, but for some cases they are useful.
		// You can throw ComLight::Exception from constructor, or from FinalConstruct() method, the library will catch & return the code from the class factory function.
		// Unfortunately, for interface methods this doesn't work, the C++ parts of the library can't catch them without very complex trickery like code generation.
		// You can still use this class in methods, but you'll need to catch them manually near the API boundary or the app will crash.
		// C++ doesn't have an ABI, the framework can't catch C++ exception across the modules.
		const HRESULT m_code;

	public:

		Exception( HRESULT hr ) : runtime_error( "ComLight HRESULT exception" ), m_code( hr ) { }

		HRESULT code() const { return m_code; }
	};
}