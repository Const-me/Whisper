using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	struct sModelSetup
	{
		eModelImplementation impl;
		eGpuModelFlags flags;
		[MarshalAs( UnmanagedType.LPWStr )]
		string? adapter;

		public sModelSetup( eGpuModelFlags flags, eModelImplementation impl, string? adapter )
		{
			this.impl = impl;
			this.flags = flags;
			this.adapter = adapter;
		}
	}

	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate void pfnListAdapters( [In, MarshalAs( UnmanagedType.LPWStr )] string name, IntPtr pv );
}