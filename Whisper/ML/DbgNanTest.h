#pragma once

namespace DirectCompute
{
	class DbgNanTest
	{
		CComPtr<ID3D11Buffer> bufferDefault, bufferStaging;
		CComPtr<ID3D11UnorderedAccessView> uav;
	public:
		HRESULT create();
		void destroy();
		operator ID3D11UnorderedAccessView* ( ) const
		{
			return uav;
		}
		bool test() const;
	};

#if DBG_TEST_NAN
	const DbgNanTest& getNanTestBuffers();
#endif
}