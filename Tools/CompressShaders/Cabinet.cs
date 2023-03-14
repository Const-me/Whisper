using System.ComponentModel;
using System.Runtime.InteropServices;

namespace CompressShaders
{
	/// <summary>Lossless data compressor implemented by <c>Cabinet.dll</c> Windows component</summary>
	/// <remarks>
	/// <para>That compression API was introduced in Windows 8.0, and is the only reason why the library won’t build for Windows 7 OS.</para>
	/// <para>Whisper.dll consumes that component in runtime, to decompress these shader binaries</para>
	/// <para>If you wonder why not gzip — because the OS doesn’t include an API for that, at least not an API usable from C or C++.<br/>
	/// .NET standard library includes gzip algorithm, but we don't want Whisper.dll to depend on .NET.</para>
	/// </remarks>
	[Obsolete]
	static class Cabinet
	{
		/// <summary>Compression algorithm</summary>
		/// <seealso href="https://learn.microsoft.com/en-us/windows/win32/cmpapi/using-the-compression-api#selecting-the-compression-algorithm" />
		enum eCompressionAlgorithm: uint
		{
			MSZIP = 2,
			XPRESS = 3,
			XPRESS_HUFF = 4,
			LZMS = 5,
		}
		/// <summary>The value should match <c>constexpr DWORD compressionAlgorithm</c> constant,<br/>in <c>Whisper/D3D/shaders.cpp</c> source file</summary>
		const eCompressionAlgorithm algo = eCompressionAlgorithm.MSZIP;

		[DllImport( "Cabinet.dll", SetLastError = true )]
		static extern bool CreateCompressor( eCompressionAlgorithm Algorithm, IntPtr AllocationRoutines, out IntPtr CompressorHandle );

		[DllImport( "Cabinet.dll", SetLastError = true )]
		static extern bool CloseCompressor( IntPtr CompressorHandle );

		[DllImport( "Cabinet.dll", SetLastError = true )]
		static extern bool Compress( IntPtr CompressorHandle, [In] byte[] UncompressedData, IntPtr UncompressedDataSize, [Out] byte[] CompressedBuffer, IntPtr CompressedBufferSize, out IntPtr CompressedDataSize );

		/// <summary>Compress an array of bytes into another, smaller array of bytes</summary>
		/// <remarks>In practice, the compression ratio is about 7.1 for the shader binaries in Release configuration.</remarks>
		public static byte[] compressBuffer( byte[] src )
		{
			if( src.Length <= 0 )
				throw new ArgumentException( "The source buffer is empty" );
			IntPtr hCompressor;
			if( !CreateCompressor( algo, IntPtr.Zero, out hCompressor ) )
				throw new Win32Exception( "Unable to create the compressor" );
			try
			{
				byte[] dest = new byte[ src.Length * 2 ];
				IntPtr srcSize = new IntPtr( src.Length );
				IntPtr destSize = new IntPtr( src.Length * 2 );
				if( !Compress( hCompressor, src, srcSize, dest, destSize, out destSize ) )
					throw new Win32Exception( "Compress failed" );
				Array.Resize( ref dest, (int)destSize );
				return dest;
			}
			finally
			{
				CloseCompressor( hCompressor );
			}
		}
	}
}