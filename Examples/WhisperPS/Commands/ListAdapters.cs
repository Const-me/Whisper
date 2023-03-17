using System.Management.Automation;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Produces list of the names of the GPUs available to Direct3D 11</para>
	/// <para type="description">You can pass any of these strings to the <c>adapter</c> argument of the <c>Import-WhisperModel</c> cmdlet.</para>
	/// </summary>
	/// <example><code>Get-Adapters</code></example>

	[Cmdlet( VerbsCommon.Get, "Adapters" )]
	public sealed class ListAdapters: Cmdlet
	{
		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			string[] arr = Library.listGraphicAdapters();
			foreach( string item in arr )
				WriteObject( item );
		}
	}
}