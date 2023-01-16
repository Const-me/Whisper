using ComLight;
using System.ComponentModel;

namespace Whisper
{
	/// <summary>A model in VRAM, loaded from GGML file.</summary>
	/// <remarks>This object doesn't keep any mutable state, and can be safely used from multiple threads concurrently</remarks>
	[ComInterface( "abefb4c9-e8d8-46a3-8747-5afbadef1adb", eMarshalDirection.ToManaged ), CustomConventions( typeof( Internal.NativeLogger ) )]
	public interface iModel: IDisposable
	{
		/// <summary>Create a context to transcribe audio with this model</summary>
		/// <remarks>Don't call this method, use <see cref="ExtensionMethods.createContext(iModel)" /> instead.</remarks>
		[RetValIndex, EditorBrowsable( EditorBrowsableState.Never )]
		Internal.iContext createContextInternal();

		/// <summary>True if this model is multi-lingual</summary>
		bool isMultilingual();

		/// <summary>Retrieve integer IDs of the special tokens defined by the model</summary>
		[RetValIndex]
		SpecialTokens getSpecialTokens();

		/// <summary>Try to resolve integer token ID into string.</summary>
		/// <remarks>Don't call this method, use <see cref="ExtensionMethods.stringFromToken(iModel, int)" /> instead.</remarks>
		IntPtr stringFromTokenInternal( int id );
	}
}