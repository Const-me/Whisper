#pragma once

namespace Whisper
{
	class MfStartupRaii
	{
		uint8_t successFlags = 0;
	public:
		MfStartupRaii() = default;
		~MfStartupRaii();
		MfStartupRaii( const MfStartupRaii& ) = delete;

		HRESULT startup();
	};
}