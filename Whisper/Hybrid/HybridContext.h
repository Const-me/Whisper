#pragma once
#include "../Whisper/WhisperModel.h"
#include "../CPU/MlContext.h"
#include "../CPU/BufferAllocator.h"
#include "KeyValueDownloader.h"
#include "../CPU/KvTensors.h"

// This version of the hybrid context uses the new, custom-built kernels
class HybridContext
{
	CpuCompute::MlContext ml;
	CpuCompute::VirtualAllocator allocCompute, allocComputeLayer;
	
	class AllocSingle : public CpuCompute::iArenaAllocator
	{
		CpuCompute::LargeBuffer buffer;
		size_t capacity = 0;
		bool allocated = false;
		// Inherited via iArenaAllocator
		virtual void* allocate( size_t cb, size_t align ) override final;

	public:
		virtual void resetArena() override final;
	};
	AllocSingle allocLayerOutput;

	const CpuCompute::DecoderTensors& model;
	const Whisper::WhisperModel& whisperModel;
	KeyValueDownloader kvCross;
	CpuCompute::KvTensors kv;

	class SetAllocatorRaii;

public:

	HybridContext( const Whisper::WhisperModel& wm );

	HRESULT create();

	HRESULT downloadKeyValues( const DirectCompute::KeyValueBuffers& source )
	{
		return kvCross.download( source );
	}

	struct sDecParams
	{
		int n_threads;
		int M;
	};

	HRESULT decode( const int* tokens, const int n_tokens, const int n_past, const sDecParams& dp, std::vector<float>& probs_out );
};