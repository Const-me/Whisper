using System;
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
		public string adapter { get; set; }

		[Parameter( Mandatory = false )]
		public eGpuModelFlags flags { get; set; }

		protected override void BeginProcessing()
		{
			if( !Environment.Is64BitProcess )
				throw new PSNotSupportedException( "Whisper cmdlets require 64-bit version of PowerShell " );
			if( !File.Exists( path ) )
				throw new FileNotFoundException( "Model file not found" );
			base.BeginProcessing();
		}

		protected override void ProcessRecord()
		{
			using( var log = this.setupLog() )
			{
				iMediaFoundation mf = Library.initMediaFoundation();
				iModel model = Library.loadModel( path, flags, adapter );
				WriteObject( new Model( mf, model ) );
			}
		}
	}
}