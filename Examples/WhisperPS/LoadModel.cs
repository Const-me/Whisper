using System.IO;
using System.Management.Automation;

namespace Whisper
{
	[Cmdlet( VerbsData.Import, "WhisperModel" )]
	public sealed class LoadModel: Cmdlet
	{
		[Parameter( Mandatory = true )]
		public string path { get; set; }

		[Parameter( Mandatory = false )]
		public string gpu { get; set; }

		[Parameter( Mandatory = false )]
		public eGpuModelFlags flags { get; set; }

		protected override void BeginProcessing()
		{
			if( !File.Exists( path ) )
				throw new FileNotFoundException( "Model file not found" );
			base.BeginProcessing();
		}

		protected override void ProcessRecord()
		{
			using( var log = this.setupLog() )
			{
				iModel model = Library.loadModel( path, flags, gpu );
				WriteObject( new Model( model ) );
			}
		}
	}
}