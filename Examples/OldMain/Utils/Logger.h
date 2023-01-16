#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_tensor;

void logError( const char8_t* pszFormat, ... );
void logWarning( const char8_t* pszFormat, ... );
void logInfo( const char8_t* pszFormat, ... );
void logDebug( const char8_t* pszFormat, ... );

#ifdef  __cplusplus
}

namespace Tracing
{
	struct ItemName
	{
		ItemName( const char* str ) { }
		ItemName( const char* str, uint32_t a0 ) { }
		ItemName( const char* str, int a0 ) { }
	};

	inline void tensor( const ItemName& name, const ggml_tensor* tensor ) { }
	inline void delayTensor( const ItemName& name, const ggml_tensor* tensor ) { }
	inline void vector( const ItemName& name, const std::vector<float>& vec ) { }
	inline void writeDelayedTensors() { }
}
#endif