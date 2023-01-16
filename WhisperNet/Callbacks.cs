using Whisper.Internal;

namespace Whisper
{
	/// <summary>Implement this abstract class to receive callbacks from the native code</summary>
	public abstract class Callbacks
	{
		/// <summary>The callback is called before every encoder run.</summary>
		/// <remarks>If it returns false, the processing is aborted.</remarks>
		protected virtual bool onEncoderBegin( Context sender ) { return true; }

		/// <summary>This callback is called on each new segment</summary>
		protected virtual void onNewSegment( Context sender, int countNew ) { }

		const int S_OK = 0;
		const int S_FALSE = 1;
		internal int encoderBegin( Context sender )
		{
			try
			{
				return onEncoderBegin( sender ) ? S_OK : S_FALSE;
			}
			catch( Exception ex )
			{
				NativeLogger.captureException( ex );
				return ex.HResult;
			}
		}

		internal int newSegment( Context sender, int countNew )
		{
			try
			{
				onNewSegment( sender, countNew );
				return S_OK;
			}
			catch( Exception ex )
			{
				NativeLogger.captureException( ex );
				return ex.HResult;
			}
		}
	}
}