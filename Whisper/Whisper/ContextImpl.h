#pragma once
#include "../API/iContext.cl.h"
#include "../ComLightLib/comLightServer.h"
#include "WhisperContext.h"
#include "Spectrogram.h"
#include "TranscribeResult.h"
#include "sTokenData.h"
#include "../ML/Device.h"

namespace Whisper
{
	class ContextImpl : public ComLight::ObjectRoot<iContext>
	{
		const DirectCompute::Device& device;
		const WhisperModel& model;
		ComLight::CComPtr<iModel> modelPtr;
		DirectCompute::WhisperContext context;
		Spectrogram spectrogram;
		int64_t mediaTimeOffset = 0;
		iSpectrogram* currentSpectrogram = nullptr;
		class CurrentSpectrogramRaii;
		ProfileCollection profiler;

		HRESULT COMLIGHTCALL getModel( iModel** pp ) override final;
		HRESULT COMLIGHTCALL timingsPrint() override final;
		HRESULT COMLIGHTCALL timingsReset() override final;
		HRESULT COMLIGHTCALL fullDefaultParams( eSamplingStrategy strategy, sFullParams* rdi ) override final;
		HRESULT COMLIGHTCALL runFullImpl( const sFullParams& params, const sProgressSink& progress, iSpectrogram& mel );
		HRESULT COMLIGHTCALL runFull( const sFullParams& params, const iAudioBuffer* buffer ) override final;
		HRESULT COMLIGHTCALL runStreamed( const sFullParams& params, const sProgressSink& progress, const iAudioReader* reader ) override final;
		HRESULT COMLIGHTCALL runCapture( const sFullParams& params, const sCaptureCallbacks& callbacks, const iAudioCapture* reader ) override final;

		struct Segment
		{
			int64_t t0;
			int64_t t1;
			mutable std::string text;
			std::vector<sTokenData> tokens;
			size_t memoryUsage() const;
		};
		std::vector<Segment> result_all;

		std::vector<whisper_token> prompt_past;

		// [EXPERIMENTAL] token-level timestamps data
		int64_t t_beg = 0;
		int64_t t_last = 0;
		whisper_token tid_last = 0;
		std::vector<float> energy; // PCM signal energy

		// [EXPERIMENTAL] speed-up techniques
		int32_t exp_n_audio_ctx = 0; // 0 - use default

		HRESULT encode( iSpectrogram& mel, int seek );
		HRESULT decode( const int* tokens, size_t length, int n_past, int threads );
		sTokenData sampleBest( const float* probs, bool force_timestamp, bool is_initial );
		sTokenData sampleBest();
		sTokenData sampleTimestamp( bool initial );
		int wrapSegment( int max_len );
		void expComputeTokenLevelTimestamps( int i_segment, float thold_pt, float thold_ptsum );

		std::vector<float> probs;
		std::vector<std::pair<double, Vocabulary::id>> probs_id;

		mutable TranscribeResultStatic results;

		HRESULT COMLIGHTCALL makeResults( eResultFlags flags, TranscribeResult& res, bool moveStrings ) const noexcept;

		HRESULT COMLIGHTCALL getResults( eResultFlags flags, iTranscribeResult** pp ) const noexcept override final;
		HRESULT COMLIGHTCALL detectSpeaker( const sTimeInterval& time, eSpeakerChannel& result ) const noexcept override final;

		int defaultThreadsCount() const;

		__m128i getMemoryUse() const;
		mutable std::vector<StereoSample> diarizeBuffer;

	public:

		ContextImpl( const DirectCompute::Device& dev, const WhisperModel& modelData, iModel* modelPointer );
	};
}