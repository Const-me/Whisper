using System;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">These flags affect compute shaders performance (which ones are faster depends on GPU model),<br/>
	/// and VRAM memory usage (UseReshapedMatMul needs slightly more VRAM).</para>
	/// </summary>
	[Flags]
	public enum eGpuModelFlags: uint
	{
		/// <summary>Equivalent to <c>Wave32 | NoReshapedMatMul</c> on Intel and nVidia GPUs,<br/>
		/// and <c>Wave64 | UseReshapedMatMul</c> on AMD GPUs</summary>
		None = 0,

		/// <summary>Use Wave32 version of compute shaders even on AMD GPUs</summary>
		/// <remarks>Incompatible with <see cref="Wave64" /></remarks>
		Wave32 = 1,

		/// <summary>Use Wave64 version of compute shaders even on nVidia and Intel GPUs</summary>
		/// <remarks>Incompatible with <see cref="Wave32" /></remarks>
		Wave64 = 2,

		/// <summary>Do not use reshaped matrix multiplication shaders on AMD GPUs</summary>
		/// <remarks>Incompatible with <see cref="UseReshapedMatMul" /></remarks>
		NoReshapedMatMul = 4,

		/// <summary>Use reshaped matrix multiplication shaders even on nVidia and Intel GPUs</summary>
		/// <remarks>Incompatible with <see cref="NoReshapedMatMul" /></remarks>
		UseReshapedMatMul = 8,

		/// <summary>Create GPU tensors in a way which allows sharing across D3D devices</summary>
		Cloneable = 0x10,
	}
}