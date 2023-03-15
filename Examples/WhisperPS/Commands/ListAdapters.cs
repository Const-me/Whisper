using System.Management.Automation;

namespace Whisper
{
	[Cmdlet( VerbsCommon.Get, "Adapters" )]
	public sealed class ListAdapters: Cmdlet
	{
		protected override void ProcessRecord()
		{
			string[] arr = Library.listGraphicAdapters();
			foreach( string item in arr )
				WriteObject( item );
		}
	}
}