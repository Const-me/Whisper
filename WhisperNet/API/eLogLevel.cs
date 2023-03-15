using System;

namespace Whisper
{
	/// <summary>Message log level</summary>
	public enum eLogLevel: byte
	{
		/// <summary>Error message</summary>
		Error = 0,
		/// <summary>Warning message</summary>
		Warning = 1,
		/// <summary>Informational message</summary>
		Info = 2,
		/// <summary>Debug message</summary>
		Debug = 3
	}

	/// <summary>A delegate to receive log messages from the library</summary>
	public delegate void pfnLogMessage( eLogLevel level, string message );

	/// <summary>Log destination flags</summary>
	[Flags]
	public enum eLoggerFlags: byte
	{
		/// <summary>No special flags</summary>
		None = 0,

		/// <summary>In addition to calling the delegate, print messaged to standard error</summary>
		UseStandardError = 1,

		/// <summary>Don’t format error codes into messages</summary>
		/// <remarks>It’s recommended to use this flag in .NET.<br/>
		/// The standard library already formats these messages automatically, as needed.</remarks>
		SkipFormatMessage = 2,
	}
}