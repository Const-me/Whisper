using System;

namespace Whisper
{
	/// <summary>Flags for <see cref="Context.results(eResultFlags)" /> method</summary>
	[Flags]
	public enum eResultFlags: uint
	{
		/// <summary>No flags</summary>
		None = 0,

		/// <summary>Return individual tokens in addition to the segments</summary>
		Tokens = 1,

		/// <summary>Return timestamps</summary>
		Timestamps = 2,

		/// <summary>Create a new COM object for the results.</summary>
		/// <remarks>Without this flag, the context returns a pointer to the COM object stored in the context.<br/>
		/// The content of that object is replaced every time you call <see cref="Internal.iContext.getResults(eResultFlags)" /> method.</remarks>
		NewObject = 0x100,
	}
}