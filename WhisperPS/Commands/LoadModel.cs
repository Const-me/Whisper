using System;
using System.IO;
using System.Management.Automation;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Load Whisper model from GGML binary file</para>
	/// <para type="description">The command might take a while to complete, depending on the disk speed.</para>
	/// </summary>
	/// <example>
	/// <code>$m = Import-WhisperModel -path D:\Data\Whisper\ggml-medium.bin</code>
	/// </example>
	/// <para type="link" uri="https://huggingface.co/datasets/ggerganov/whisper.cpp">Download models from Hugging Face</para>

	[Cmdlet( VerbsData.Import, "WhisperModel" )]
	public sealed class LoadModel: PSCmdlet
	{
		/// <summary>
		/// <para type="synopsis">Path to the GGML file on disk</para>
		/// </summary>
		[Parameter( Mandatory = true, Position = 0 ), ValidateNotNullOrEmpty]
		public string path { get; set; }

		/// <summary>
		/// <para type="synopsis">Optional name of the graphics adapter to use</para>
		/// <para type="description">Use <c>Get-Adapters</c> command to list available graphics adapter in this computer</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public string adapter { get; set; }

		/// <summary>
		/// <para type="synopsis">Advanced GPU flags</para>
		/// <para type="description">To combine multiple flags, use comma for the separator, example:<br />
		/// <c>-flags UseReshapedMatMul, Wave64</c></para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public eGpuModelFlags flags { get; set; }

		protected override void BeginProcessing()
		{
			if( !Environment.Is64BitProcess )
				throw new PSNotSupportedException( "Whisper cmdlets require 64-bit version of PowerShell " );
			base.BeginProcessing();
		}

		void reportProgress( double progressValue ) =>
			this.writeProgress( progressValue, "Loading model" );

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			string path = this.absolutePath( this.path );
			if( !File.Exists( path ) )
				throw new FileNotFoundException( "Model file not found" );

			using( var log = this.setupLog() )
			{
				iMediaFoundation mf = Library.initMediaFoundation();
				iModel model = Library.loadModel( path, flags, adapter, reportProgress );
				WriteObject( new Model( mf, model ) );
			}
		}
	}
}