using Whisper.Internal;

namespace Whisper
{
	/// <summary>Implement this abstract class to provide callbacks for audio capture method</summary>
	public abstract class CaptureCallbacks
	{
		/// <summary>Override this method to support cancellation</summary>
		protected virtual bool shouldCancel( Context sender ) { return false; }

		/// <summary>Override this method to get notified about status changes</summary>
		protected virtual void captureStatusChanged( Context sender, eCaptureStatus status ) { }

		internal pfnShouldCancel cancel( Context sender )
		{
			const int S_OK = 0;
			const int S_FALSE = 1;
			return delegate ( IntPtr pv )
			{
				try
				{
					return shouldCancel( sender ) ? S_FALSE : S_OK;
				}
				catch( Exception ex )
				{
					NativeLogger.captureException( ex );
					return ex.HResult;
				}
			};
		}

		internal pfnCaptureStatus status( Context sender )
		{
			return delegate ( IntPtr pv, eCaptureStatus status )
			{
				try
				{
					captureStatusChanged( sender, status );
					return 0;
				}
				catch( Exception ex )
				{
					NativeLogger.captureException( ex );
					return ex.HResult;
				}
			};
		}
	}
}