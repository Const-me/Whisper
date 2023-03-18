using System;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">This object holds a Whisper model, loaded from disk to VRAM on the GPU.</para>
	/// <para type="description">For large models, the data size may exceed 4GB of video memory</para>
	/// </summary>
	public sealed class Model: IDisposable
	{
		internal iMediaFoundation mf { get; private set; }
		internal iModel model { get; private set; }

		internal Model( iMediaFoundation mf, iModel model )
		{
			this.mf = mf;
			this.model = model;
		}

		public void Dispose()
		{
			mf?.Dispose();
			mf = null;
			model?.Dispose();
			model = null;
			GC.SuppressFinalize( this );
		}

		~Model()
		{
			mf?.Dispose();
			mf = null;
			model?.Dispose();
			model = null;
		}
	}
}