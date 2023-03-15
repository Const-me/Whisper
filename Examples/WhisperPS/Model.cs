using System;

namespace Whisper
{
	public sealed class Model: IDisposable
	{
		iModel model;

		internal Model( iModel model )
		{
			this.model = model;
		}

		public void Dispose()
		{
			model?.Dispose();
			model = null;
			GC.SuppressFinalize( this );
		}

		~Model()
		{
			model?.Dispose();
			model = null;
		}
	}
}