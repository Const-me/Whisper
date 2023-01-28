namespace Whisper
{
	/// <summary>Output value for iContext.detectSpeaker method</summary>
	public enum eSpeakerChannel: byte
	{
		/// <summary>Unable to detect</summary>
		Unsure = 0,
		/// <summary>The speech was mostly in the left channel</summary>
		Left = 1,
		/// <summary>The speech was mostly in the right channel</summary>
		Right = 2,
		/// <summary>The audio only has 1 channel</summary>
		NoStereoData = 0xFF,
	}
}