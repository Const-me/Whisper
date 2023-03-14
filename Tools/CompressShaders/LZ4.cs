namespace CompressShaders;
using K4os.Compression.LZ4;

/// <summary>Lossless data compressor which uses LZ4-HC compressor</summary>
/// <seealso href="https://github.com/lz4/lz4" />
/// <seealso href="https://github.com/MiloszKrajewski/K4os.Compression.LZ4" />
static class LZ4
{
	// compression speed drops rapidly when not using FAST mode, while decompression speed stays the same
	// Actually, it is usually faster for high compression levels as there is less data to process
	// https://github.com/MiloszKrajewski/K4os.Compression.LZ4#compression-levels
	const LZ4Level compressionLevel = LZ4Level.L12_MAX;

	public static byte[] compressBuffer( byte[] src )
	{
		int maxLength = LZ4Codec.MaximumOutputSize( src.Length );
		byte[] output = new byte[ maxLength ];
		int cb = LZ4Codec.Encode( src, output, compressionLevel );
		if( cb > 0 )
		{
			Array.Resize( ref output, cb );
			return output;
		}
		throw new ApplicationException( $"LZ4Codec.Encode failed with status {cb}" );
	}
}