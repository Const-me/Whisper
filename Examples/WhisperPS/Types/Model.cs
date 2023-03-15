using System;

namespace Whisper
{
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