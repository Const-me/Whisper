#pragma once

namespace DirectCompute
{
	bool initializeRenderDoc();

	class CaptureRaii
	{
		bool capturing;
	public:
		CaptureRaii();
		CaptureRaii( const CaptureRaii& ) = delete;
		~CaptureRaii();
	};
}