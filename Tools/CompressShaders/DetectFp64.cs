#pragma warning disable CS0649
using System.Runtime.InteropServices;

namespace CompressShaders
{
	static class DetectFp64
	{
		struct DXBCHeader
		{
			public uint FourCC; // Four character code "DXBC"
			public uint Hash0;            // 32-bit hash of the DXBC file
			public uint Hash1;            // 32-bit hash of the DXBC file
			public uint Hash2;            // 32-bit hash of the DXBC file
			public uint Hash3;            // 32-bit hash of the DXBC file
			public uint unknownOne;
			public uint TotalSize;        // Total size of the DXBC file in bytes
			public int NumChunks;        // Number of chunks in the DXBC file
		};

		public static bool usesFp64( ReadOnlySpan<byte> dxbc )
		{
			ReadOnlySpan<DXBCHeader> dxbcHeaderSpan = MemoryMarshal.Cast<byte, DXBCHeader>( dxbc );
			DXBCHeader dxbcHeader = dxbcHeaderSpan[ 0 ];

			int cbHeader = Marshal.SizeOf<DXBCHeader>();
			int nChunks = dxbcHeader.NumChunks;
			ReadOnlySpan<int> chunkOffsets = MemoryMarshal.Cast<byte, int>( dxbc.Slice( cbHeader, nChunks * 4 ) );
			foreach( int off in chunkOffsets )
			{
				uint id = MemoryMarshal.Cast<byte, uint>( dxbc.Slice( off, 4 ) )[ 0 ];
				const uint SFI0 = 0x30494653;
				if( id != SFI0 )
					continue;
				int size = MemoryMarshal.Cast<byte, int>( dxbc.Slice( off + 4, 4 ) )[ 0 ];
				if( size < 4 )
					throw new ApplicationException();
				uint data = MemoryMarshal.Cast<byte, uint>( dxbc.Slice( off + 8, 4 ) )[ 0 ];
				return 0 != ( data & 1u );
			}
			return false;
		}
	}
}