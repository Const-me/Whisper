#include "stdafx.h"
#include "ML/Tensor.h"
#include "API/iMediaFoundation.cl.h"
#include "API/iContext.cl.h"
#include "API/sFullParams.h"
#include "Utils/ReadStream.h"
#include "ML/testUtils.h"
#include "Utils/Trace/tracing.h"
#include "modelFactory.h"
#if BUILD_BOTH_VERSIONS
#ifndef __AVX__
#error Reference version requires AVX build, and AVX2 CPU
#endif // !__AVX__

namespace
{
	LPCTSTR traceFilePath = LR"(C:\Temp\2remove\Whisper\ref.bin)";
	using ComLight::iReadStream;
}

struct whisper_context;
struct ggml_tensor;

class GpuEncTest
{
	DirectCompute::Tensor mel, gpuResult;

	DirectCompute::Tensor tempGpu;
	const ggml_tensor* tempRef = nullptr;
public:
	GpuEncTest( const whisper_context& wctx, const int mel_offset );
	void compare( const ggml_tensor* expected ) const;
	void compareMel( const ggml_tensor* expected ) const;
};

class GpuDecTest
{
	std::vector<float> logits, probs;
	const ggml_tensor* tempRef = nullptr;

public:

	GpuDecTest( const whisper_context& wctx, const int* tokens, const int n_tokens, const int n_past );

	void postpone( const ggml_tensor* t );
	void comparePostponed();
	void compare( const std::vector<float>& cpuLogits, const std::vector<float>& cpuProbs ) const;
};

static DirectCompute::Tensor gpuEncode( const whisper_context& wctx, const int mel_offset );

#include "source/whisper.cpp"
#include "API/iContext.cl.h"
#include "../ComLightLib/comLightServer.h"
#include "Whisper/WhisperContext.h"
#include "Whisper/ModelLoader.h"
#include "Whisper/WhisperModel.h"
#include "source.compat/convertThings.h"

namespace Whisper
{
	inline HRESULT isZero( int i )
	{
		return ( 0 == i ) ? S_OK : E_FAIL;
	}

	class Context : public ComLight::ObjectRoot<iContext>,
		public iModel
	{
		virtual HRESULT COMLIGHTCALL isMultilingual() override final
		{
			return whisper_is_multilingual( &ctx ) ? S_OK : S_FALSE;
		}
		virtual const char* COMLIGHTCALL stringFromToken( whisper_token token ) override final
		{
			return whisper_token_to_str( &ctx, token );
		}
		virtual HRESULT COMLIGHTCALL getSpecialTokens( SpecialTokens& rdi )
		{
			rdi.TranscriptionEnd = whisper_token_eot( &ctx );
			rdi.TranscriptionStart = whisper_token_sot( &ctx );
			rdi.PreviousWord = whisper_token_prev( &ctx );
			rdi.SentenceStart = whisper_token_solm( &ctx );
			rdi.Not = whisper_token_not( &ctx );
			rdi.TranscriptionBegin = whisper_token_beg( &ctx );
			rdi.TaskTranslate = whisper_token_translate();
			rdi.TaskTranscribe = whisper_token_transcribe();
			return S_OK;
		}
		HRESULT COMLIGHTCALL tokenize( const char* text, pfnDecodedTokens pfn, void* pv ) override final
		{
			const auto res = ::tokenize( ctx.vocab, text );
			if( !res.empty() )
				pfn( res.data(), res.size(), pv );
			return S_OK;
		}
		HRESULT COMLIGHTCALL clone( iModel** rdi ) override final
		{
			logError( u8"Reference CPU model doesn’t support clone()" );
			return E_NOTIMPL;
		}
		HRESULT COMLIGHTCALL detectSpeaker( const sTimeInterval& time, eSpeakerChannel& result ) const override final
		{
			logError( u8"Reference CPU model doesn’t support speaker detection" );
			return E_NOTIMPL;
		}

		// Performance information
		virtual HRESULT COMLIGHTCALL timingsPrint() override final
		{
			whisper_print_timings( &ctx );
			return S_OK;
		}
		virtual HRESULT COMLIGHTCALL timingsReset() override final
		{
			whisper_reset_timings( &ctx );
			return S_OK;
		}

		virtual HRESULT COMLIGHTCALL fullDefaultParams( eSamplingStrategy strategy, sFullParams* rdi )
		{
			static_assert( (int)eSamplingStrategy::Greedy == whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY );
			static_assert( (int)eSamplingStrategy::BeamSearch == whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH );
			const whisper_sampling_strategy wss = (whisper_sampling_strategy)(int)strategy;
			whisper_full_params wfp = whisper_full_default_params( wss );

			*rdi = makeNewParams( wfp );
			return S_OK;
		}

		HRESULT COMLIGHTCALL runFull( const sFullParams& params, const iAudioBuffer* buffer ) override final
		{
			whisper_full_params wfp = makeOldParams( params, this );
			const float* const samples = buffer->getPcmMono();
			const uint32_t n_samples = buffer->countSamples();
			return isZero( whisper_full( &ctx, wfp, samples, (int)n_samples ) );
		}

		HRESULT COMLIGHTCALL runStreamed( const sFullParams& params, const sProgressSink& progress, const iAudioReader* reader ) override final
		{
			logError( u8"The CPU reference implementation doesn’t support streaming" );
			return E_NOTIMPL;
		}
		HRESULT COMLIGHTCALL runCapture( const sFullParams& params, const sCaptureCallbacks& callbacks, const iAudioCapture* reader ) override final
		{
			logError( u8"The CPU reference implementation doesn’t support audio capture" );
			return E_NOTIMPL;
		}

		HRESULT COMLIGHTCALL getResults( eResultFlags flags, iTranscribeResult** pp ) const override final
		{
			makeNewResults( &ctx, flags, pp );
			return S_OK;
		}

		HRESULT loadImpl( iReadStream* stm );

		virtual HRESULT COMLIGHTCALL createContext( iContext** pp ) override final
		{
			if( nullptr == pp )
				return E_POINTER;
			*pp = this;
			( *pp )->AddRef();
			return S_OK;
		}

		virtual HRESULT COMLIGHTCALL getModel( iModel** pp ) override final
		{
			if( nullptr == pp )
				return E_POINTER;
			*pp = this;
			( *pp )->AddRef();
			return S_OK;
		}

	public:

		Context()
		{
			if( nullptr != traceFilePath )
				Tracing::traceCreate( traceFilePath );
		}

		mutable whisper_context ctx;

		HRESULT load( iReadStream* stm );

		~Context()
		{
			Tracing::traceClose();

			if( ctx.model.ctx )
			{
				ggml_free( ctx.model.ctx );
				ctx.model.ctx = nullptr;
			}
			if( ctx.model.ctx_mem )
			{
				ggml_free( ctx.model.ctx_mem );
				ctx.model.ctx_mem = nullptr;
			}
			if( ctx.buf_model )
			{
				delete ctx.buf_model;
				ctx.buf_model = nullptr;
			}
		}

		BEGIN_COM_MAP()
			COM_INTERFACE_ENTRY( iModel );
		END_COM_MAP()
	};

	inline HRESULT readBytes( iReadStream* stm, void* rdi, size_t cb )
	{
		if( cb > INT_MAX )
			return DISP_E_OVERFLOW;
		if( cb == 0 )
			return S_FALSE;
		int n;
		CHECK( stm->read( rdi, (int)cb, n ) );
		if( n != (int)cb )
			return E_EOF;
		return S_OK;
	}

	template<typename T>
	inline HRESULT readStruct( iReadStream* stm, T& dest )
	{
		return readBytes( stm, &dest, sizeof( T ) );
	}
	template<typename E>
	inline HRESULT readVector( iReadStream* stm, std::vector<E>& vec )
	{
		const size_t cb = sizeof( E ) * vec.size();
		if( cb > 0 )
			return readBytes( stm, vec.data(), cb );
		return S_FALSE;
	}

	inline HRESULT readString( iReadStream* stm, std::string& str )
	{
		uint32_t len;
		CHECK( readStruct( stm, len ) );
		if( len > 0 )
		{
			str.resize( len );
			return readBytes( stm, str.data(), len );
		}
		else
		{
			str.clear();
			return S_FALSE;
		}
	}

	// load the model from a ggml file
	// file format:
	//   - hparams
	//   - pre-computed mel filters
	//   - vocab
	//   - weights
	// see the convert-pt-to-ggml.py script for details
	HRESULT Context::loadImpl( iReadStream* stm )
	{
		// WhisperModel wm;
		// return wm.load( stm );

		// Copy-pasted from whisper_model_load() function
		auto& model = ctx.model;
		auto& vocab = ctx.vocab;

		// verify magic
		{
			uint32_t magic;
			int cbRead;
			CHECK( stm->read( &magic, 4, cbRead ) );
			if( magic != 0x67676d6c )
			{
				logError( u8"Invalid model file, bad magic" );
				return E_INVALIDARG;
			}
		}

		//load hparams
		{
			auto& hparams = model.hparams;
			CHECK( readStruct( stm, hparams ) );
			assert( hparams.n_text_state == hparams.n_audio_state );

			if( hparams.n_audio_layer == 4 )
				model.type = e_model::MODEL_TINY;
			if( hparams.n_audio_layer == 6 )
				model.type = e_model::MODEL_BASE;
			if( hparams.n_audio_layer == 12 )
				model.type = e_model::MODEL_SMALL;
			if( hparams.n_audio_layer == 24 )
				model.type = e_model::MODEL_MEDIUM;
			if( hparams.n_audio_layer == 32 )
				model.type = e_model::MODEL_LARGE;

			logDebug( u8"%s: n_vocab       = %d", __func__, hparams.n_vocab );
			logDebug( u8"%s: n_audio_ctx   = %d", __func__, hparams.n_audio_ctx );
			logDebug( u8"%s: n_audio_state = %d", __func__, hparams.n_audio_state );
			logDebug( u8"%s: n_audio_head  = %d", __func__, hparams.n_audio_head );
			logDebug( u8"%s: n_audio_layer = %d", __func__, hparams.n_audio_layer );
			logDebug( u8"%s: n_text_ctx    = %d", __func__, hparams.n_text_ctx );
			logDebug( u8"%s: n_text_state  = %d", __func__, hparams.n_text_state );
			logDebug( u8"%s: n_text_head   = %d", __func__, hparams.n_text_head );
			logDebug( u8"%s: n_text_layer  = %d", __func__, hparams.n_text_layer );
			logDebug( u8"%s: n_mels        = %d", __func__, hparams.n_mels );
			logDebug( u8"%s: f16           = %d", __func__, hparams.f16 );
			logDebug( u8"%s: type          = %d", __func__, model.type );

			ctx.buf_model = new std::vector<uint8_t>();
			ctx.buf_model->resize( MEM_REQ_MODEL.at( model.type ) );
			ctx.buf_memory.resize( MEM_REQ_MEMORY.at( model.type ) );
			ctx.buf_compute.resize( std::max( MEM_REQ_ENCODE.at( model.type ), MEM_REQ_DECODE.at( model.type ) ) );
			ctx.buf_compute_layer.resize( std::max( MEM_REQ_ENCODE_LAYER.at( model.type ), MEM_REQ_DECODE_LAYER.at( model.type ) ) );
		}

		// load mel filters
		{
			auto& filters = ctx.model.filters;
			CHECK( readStruct( stm, filters.n_mel ) );
			CHECK( readStruct( stm, filters.n_fft ) );
			filters.data.resize( filters.n_mel * filters.n_fft );
			CHECK( readVector( stm, filters.data ) );
		}

		// load vocab
		{
			int32_t n_vocab = 0;
			CHECK( readStruct( stm, n_vocab ) );

			//if (n_vocab != model.hparams.n_vocab) {
			//    fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
			//            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
			//    return false;
			//}

			std::string word;
			for( int i = 0; i < n_vocab; i++ )
			{
				CHECK( readString( stm, word ) );
				vocab.token_to_id[ word ] = i;
				vocab.id_to_token[ i ] = word;
			}

			vocab.n_vocab = model.hparams.n_vocab;
			if( vocab.is_multilingual() )
			{
				vocab.token_eot++;
				vocab.token_sot++;
				vocab.token_prev++;
				vocab.token_solm++;
				vocab.token_not++;
				vocab.token_beg++;
			}

			if( n_vocab < model.hparams.n_vocab )
			{
				logDebug( u8"%s: adding %d extra tokens", __func__, model.hparams.n_vocab - n_vocab );
				for( int i = n_vocab; i < model.hparams.n_vocab; i++ )
				{
					if( i > vocab.token_beg )
						word = "[_TT_" + std::to_string( i - vocab.token_beg ) + "]";
					else if( i == vocab.token_eot )
						word = "[_EOT_]";
					else if( i == vocab.token_sot )
						word = "[_SOT_]";
					else if( i == vocab.token_prev )
						word = "[_PREV_]";
					else if( i == vocab.token_not )
						word = "[_NOT_]";
					else if( i == vocab.token_beg )
						word = "[_BEG_]";
					else
						word = "[_extra_token_" + std::to_string( i ) + "]";

					vocab.token_to_id[ word ] = i;
					vocab.id_to_token[ i ] = word;
				}
			}
		}

		{
			// this is the total memory required to run the inference
			const size_t mem_required =
				ctx.buf_model->size() +
				ctx.buf_memory.size() +
				ctx.buf_compute.size() +
				ctx.buf_compute_layer.size();
			logDebug( u8"%s: mem_required  = %7.2f MB", __func__, mem_required / 1024.0 / 1024.0 );
		}

		// for the big tensors, we have the option to store the data in 16-bit floats
		// in order to save memory and also to speed up the computation
		const ggml_type wtype = model.hparams.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

		size_t ctx_size = 0;
		size_t ctx_mem_size = 0;

		{
			const auto& hparams = model.hparams;

			const int n_vocab = hparams.n_vocab;

			const int n_audio_ctx = hparams.n_audio_ctx;
			const int n_audio_state = hparams.n_audio_state;
			const int n_audio_layer = hparams.n_audio_layer;

			const int n_text_ctx = hparams.n_text_ctx;
			const int n_text_state = hparams.n_text_state;
			const int n_text_layer = hparams.n_text_layer;

			const int n_mels = hparams.n_mels;

			// encoder
			{
				// TODO: F16 .. maybe not?
				ctx_size += n_audio_ctx * n_audio_state * ggml_type_size( GGML_TYPE_F32 ); // e_pe;

				ctx_size += 3 * n_mels * n_audio_state * ggml_type_size( wtype );         // e_conv_1_w
				ctx_size += n_audio_state * ggml_type_size( GGML_TYPE_F32 ); // e_conv_1_b

				ctx_size += 3 * n_audio_state * n_audio_state * ggml_type_size( wtype );         // e_conv_2_w
				ctx_size += n_audio_state * ggml_type_size( GGML_TYPE_F32 ); // e_conv_2_b

				ctx_size += n_audio_state * ggml_type_size( GGML_TYPE_F32 ); // e_ln_w;
				ctx_size += n_audio_state * ggml_type_size( GGML_TYPE_F32 ); // e_ln_b;
			}

			// decoder
			{
				// TODO: F16 .. maybe not?
				ctx_size += n_text_ctx * n_text_state * ggml_type_size( GGML_TYPE_F32 ); // d_pe;

				ctx_size += n_vocab * n_text_state * ggml_type_size( wtype ); // d_te;

				ctx_size += n_text_state * ggml_type_size( GGML_TYPE_F32 ); // d_ln_w;
				ctx_size += n_text_state * ggml_type_size( GGML_TYPE_F32 ); // d_ln_b;
			}

			// encoder layers
			{
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_ln_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_ln_b

				ctx_size += n_audio_layer * ( 4 * n_audio_state * n_audio_state * ggml_type_size( wtype ) );         // mlp_0_w
				ctx_size += n_audio_layer * ( 4 * n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_0_b

				ctx_size += n_audio_layer * ( 4 * n_audio_state * n_audio_state * ggml_type_size( wtype ) );         // mlp_1_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_1_b

				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_0_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_0_b

				ctx_size += n_audio_layer * ( n_audio_state * n_audio_state * ggml_type_size( wtype ) );         // attn_q_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_q_b

				ctx_size += n_audio_layer * ( n_audio_state * n_audio_state * ggml_type_size( wtype ) ); // attn_k_w

				ctx_size += n_audio_layer * ( n_audio_state * n_audio_state * ggml_type_size( wtype ) );         // attn_v_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_v_b

				ctx_size += n_audio_layer * ( n_audio_state * n_audio_state * ggml_type_size( wtype ) );         // attn_ln_1_w
				ctx_size += n_audio_layer * ( n_audio_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_1_b
			}

			// decoder layers
			{
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_ln_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_ln_b

				ctx_size += n_text_layer * ( 4 * n_text_state * n_text_state * ggml_type_size( wtype ) );         // mlp_0_w
				ctx_size += n_text_layer * ( 4 * n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_0_b

				ctx_size += n_text_layer * ( 4 * n_text_state * n_text_state * ggml_type_size( wtype ) );         // mlp_1_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // mlp_1_b

				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_0_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_0_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // attn_q_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_q_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) ); // attn_k_w

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // attn_v_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_v_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // attn_ln_1_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // attn_ln_1_b
				//
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // cross_attn_ln_0_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // cross_attn_ln_0_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // cross_attn_q_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // cross_attn_q_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) ); // cross_attn_k_w

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // cross_attn_v_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // cross_attn_v_b

				ctx_size += n_text_layer * ( n_text_state * n_text_state * ggml_type_size( wtype ) );         // cross_attn_ln_1_w
				ctx_size += n_text_layer * ( n_text_state * ggml_type_size( GGML_TYPE_F32 ) ); // cross_attn_ln_1_b
			}

			ctx_mem_size += n_text_layer * n_text_ctx * n_text_state * ggml_type_size( GGML_TYPE_F16 ); // memory_k
			ctx_mem_size += n_text_layer * n_text_ctx * n_text_state * ggml_type_size( GGML_TYPE_F16 ); // memory_v

			ctx_mem_size += n_text_layer * n_audio_ctx * n_text_state * ggml_type_size( GGML_TYPE_F16 ); // memory_cross_k
			ctx_mem_size += n_text_layer * n_audio_ctx * n_text_state * ggml_type_size( GGML_TYPE_F16 ); // memory_cross_v

			ctx_size += ( 15 + 15 * n_audio_layer + 24 * n_text_layer ) * 256; // object overhead

			logDebug( u8"%s: ggml ctx size = %7.2f MB", __func__, ctx_size / ( 1024.0 * 1024.0 ) );
		}

		// create the ggml context
		{
			struct ggml_init_params params;
			params.mem_size = ctx.buf_model->size();
			params.mem_buffer = ctx.buf_model->data();

			model.ctx = ggml_init( params );
			if( !model.ctx )
			{
				logError( u8"%s: ggml_init() failed", __func__ );
				return E_INVALIDARG;
			}
		}

		std::map<std::string, struct ggml_tensor*> tensors;
		DirectCompute::ModelLoader loader{ model.hparams.n_audio_layer, model.hparams.n_text_layer };

		// prepare memory for the weights
		{
			auto& ctx = model.ctx;
			const auto& hparams = model.hparams;
			const int n_vocab = hparams.n_vocab;

			const int n_audio_ctx = hparams.n_audio_ctx;
			const int n_audio_state = hparams.n_audio_state;
			const int n_audio_layer = hparams.n_audio_layer;

			const int n_text_ctx = hparams.n_text_ctx;
			const int n_text_state = hparams.n_text_state;
			const int n_text_layer = hparams.n_text_layer;

			const int n_mels = hparams.n_mels;

			model.layers_encoder.resize( n_audio_layer );
			model.layers_decoder.resize( n_text_layer );

			// encoder
			{
				model.e_pe = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx );
				loader.add( model.e_pe, loader.model.enc.positionalEmbedding );

				model.e_conv_1_w = ggml_new_tensor_3d( ctx, wtype, 3, n_mels, n_audio_state );
				model.e_conv_1_b = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, n_audio_state );
				loader.add( model.e_conv_1_w, model.e_conv_1_b, loader.model.enc.conv1 );

				model.e_conv_2_w = ggml_new_tensor_3d( ctx, wtype, 3, n_audio_state, n_audio_state );
				model.e_conv_2_b = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, 1, n_audio_state );
				loader.add( model.e_conv_2_w, model.e_conv_2_b, loader.model.enc.conv2 );

				model.e_ln_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
				model.e_ln_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
				loader.add( model.e_ln_w, model.e_ln_b, loader.model.enc.lnPost );

				// map by name
				tensors[ "encoder.positional_embedding" ] = model.e_pe;

				tensors[ "encoder.conv1.weight" ] = model.e_conv_1_w;
				tensors[ "encoder.conv1.bias" ] = model.e_conv_1_b;

				tensors[ "encoder.conv2.weight" ] = model.e_conv_2_w;
				tensors[ "encoder.conv2.bias" ] = model.e_conv_2_b;

				tensors[ "encoder.ln_post.weight" ] = model.e_ln_w;
				tensors[ "encoder.ln_post.bias" ] = model.e_ln_b;

				for( int i = 0; i < n_audio_layer; ++i )
				{
					auto& layer = model.layers_encoder[ i ];
					auto& gpu = loader.model.enc.layers[ i ];

					layer.mlp_ln_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					layer.mlp_ln_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.mlp_ln_w, layer.mlp_ln_b, gpu.mlpLn );

					layer.mlp_0_w = ggml_new_tensor_2d( ctx, wtype, n_audio_state, 4 * n_audio_state );
					layer.mlp_0_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 4 * n_audio_state );
					loader.add( layer.mlp_0_w, layer.mlp_0_b, gpu.mlp0 );

					layer.mlp_1_w = ggml_new_tensor_2d( ctx, wtype, 4 * n_audio_state, n_audio_state );
					layer.mlp_1_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.mlp_1_w, layer.mlp_1_b, gpu.mlp1 );

					layer.attn_ln_0_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					layer.attn_ln_0_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.attn_ln_0_w, layer.attn_ln_0_b, gpu.attnLn0 );

					layer.attn_q_w = ggml_new_tensor_2d( ctx, wtype, n_audio_state, n_audio_state );
					layer.attn_q_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.attn_q_w, layer.attn_q_b, gpu.attnQuery );

					layer.attn_k_w = ggml_new_tensor_2d( ctx, wtype, n_audio_state, n_audio_state );
					loader.add( layer.attn_k_w, gpu.attnKey );

					layer.attn_v_w = ggml_new_tensor_2d( ctx, wtype, n_audio_state, n_audio_state );
					layer.attn_v_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.attn_v_w, layer.attn_v_b, gpu.attnValue );

					layer.attn_ln_1_w = ggml_new_tensor_2d( ctx, wtype, n_audio_state, n_audio_state );
					layer.attn_ln_1_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_audio_state );
					loader.add( layer.attn_ln_1_w, layer.attn_ln_1_b, gpu.attnLn1 );

					// map by name
					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp_ln.weight" ] = layer.mlp_ln_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp_ln.bias" ] = layer.mlp_ln_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp.0.weight" ] = layer.mlp_0_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp.0.bias" ] = layer.mlp_0_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp.2.weight" ] = layer.mlp_1_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".mlp.2.bias" ] = layer.mlp_1_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn_ln.weight" ] = layer.attn_ln_0_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn_ln.bias" ] = layer.attn_ln_0_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.query.weight" ] = layer.attn_q_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.query.bias" ] = layer.attn_q_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.key.weight" ] = layer.attn_k_w;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.value.weight" ] = layer.attn_v_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.value.bias" ] = layer.attn_v_b;

					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.out.weight" ] = layer.attn_ln_1_w;
					tensors[ "encoder.blocks." + std::to_string( i ) + ".attn.out.bias" ] = layer.attn_ln_1_b;
				}
			}

			// decoder
			{
				model.d_pe = ggml_new_tensor_2d( ctx, GGML_TYPE_F32, n_text_state, n_text_ctx );
				loader.add( model.d_pe, loader.model.dec.positionalEmbedding );

				model.d_te = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_vocab );
				loader.add( model.d_te, loader.model.dec.tokenEmbedding );

				model.d_ln_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
				model.d_ln_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
				loader.add( model.d_ln_w, model.d_ln_b, loader.model.dec.ln );

				// map by name
				tensors[ "decoder.positional_embedding" ] = model.d_pe;

				tensors[ "decoder.token_embedding.weight" ] = model.d_te;

				tensors[ "decoder.ln.weight" ] = model.d_ln_w;
				tensors[ "decoder.ln.bias" ] = model.d_ln_b;

				for( int i = 0; i < n_text_layer; ++i ) {
					auto& layer = model.layers_decoder[ i ];
					auto& gpu = loader.model.dec.layers[ i ];

					layer.mlp_ln_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					layer.mlp_ln_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.mlp_ln_w, layer.mlp_ln_b, gpu.mlpLn );

					layer.mlp_0_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, 4 * n_text_state );
					layer.mlp_0_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, 4 * n_text_state );
					loader.add( layer.mlp_0_w, layer.mlp_0_b, gpu.mlp0 );

					layer.mlp_1_w = ggml_new_tensor_2d( ctx, wtype, 4 * n_text_state, n_text_state );
					layer.mlp_1_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.mlp_1_w, layer.mlp_1_b, gpu.mlp1 );

					layer.attn_ln_0_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					layer.attn_ln_0_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.attn_ln_0_w, layer.attn_ln_0_b, gpu.attnLn0 );

					layer.attn_q_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.attn_q_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.attn_q_w, layer.attn_q_b, gpu.attnQuery );

					layer.attn_k_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					loader.add( layer.attn_k_w, gpu.attnKey );

					layer.attn_v_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.attn_v_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.attn_v_w, layer.attn_v_b, gpu.attnValue );

					layer.attn_ln_1_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.attn_ln_1_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.attn_ln_1_w, layer.attn_ln_1_b, gpu.attnLn1 );

					layer.cross_attn_ln_0_w = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					layer.cross_attn_ln_0_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.cross_attn_ln_0_w, layer.cross_attn_ln_0_b, gpu.crossAttnLn0 );

					layer.cross_attn_q_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.cross_attn_q_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.cross_attn_q_w, layer.cross_attn_q_b, gpu.crossAttnQuery );

					layer.cross_attn_k_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					loader.add( layer.cross_attn_k_w, gpu.crossAttnKey );

					layer.cross_attn_v_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.cross_attn_v_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.cross_attn_v_w, layer.cross_attn_v_b, gpu.crossAttnValue );

					layer.cross_attn_ln_1_w = ggml_new_tensor_2d( ctx, wtype, n_text_state, n_text_state );
					layer.cross_attn_ln_1_b = ggml_new_tensor_1d( ctx, GGML_TYPE_F32, n_text_state );
					loader.add( layer.cross_attn_ln_1_w, layer.cross_attn_ln_1_b, gpu.crossAttnLn1 );

					// map by name
					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp_ln.weight" ] = layer.mlp_ln_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp_ln.bias" ] = layer.mlp_ln_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp.0.weight" ] = layer.mlp_0_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp.0.bias" ] = layer.mlp_0_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp.2.weight" ] = layer.mlp_1_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".mlp.2.bias" ] = layer.mlp_1_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn_ln.weight" ] = layer.attn_ln_0_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn_ln.bias" ] = layer.attn_ln_0_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.query.weight" ] = layer.attn_q_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.query.bias" ] = layer.attn_q_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.key.weight" ] = layer.attn_k_w;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.value.weight" ] = layer.attn_v_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.value.bias" ] = layer.attn_v_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.out.weight" ] = layer.attn_ln_1_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".attn.out.bias" ] = layer.attn_ln_1_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn_ln.weight" ] = layer.cross_attn_ln_0_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn_ln.bias" ] = layer.cross_attn_ln_0_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.query.weight" ] = layer.cross_attn_q_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.query.bias" ] = layer.cross_attn_q_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.key.weight" ] = layer.cross_attn_k_w;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.value.weight" ] = layer.cross_attn_v_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.value.bias" ] = layer.cross_attn_v_b;

					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.out.weight" ] = layer.cross_attn_ln_1_w;
					tensors[ "decoder.blocks." + std::to_string( i ) + ".cross_attn.out.bias" ] = layer.cross_attn_ln_1_b;
				}
			}
		}

		// create the ggml memory context
		{
			struct ggml_init_params params;
			params.mem_size = ctx.buf_memory.size();
			params.mem_buffer = ctx.buf_memory.data();
			model.ctx_mem = ggml_init( params );
			if( !model.ctx_mem )
			{
				logError( u8"%s: ggml_init() failed", __func__ );
				return E_INVALIDARG;
			}
		}

		// key + value memory
		{
			auto& ctx = model.ctx_mem;

			const auto& hparams = model.hparams;

			const int n_text_state = hparams.n_text_state;
			const int n_text_layer = hparams.n_text_layer;
			const int n_text_ctx = hparams.n_text_ctx;

			// key/value memory for the self-attention layer
			{
				const int n_mem = n_text_layer * n_text_ctx;
				const int n_elements = n_text_state * n_mem;

				model.memory_k = ggml_new_tensor_1d( ctx, GGML_TYPE_F16, n_elements );
				model.memory_v = ggml_new_tensor_1d( ctx, GGML_TYPE_F16, n_elements );
			}

			// key/value memory for the cross-attention layer
			{
				const int n_audio_ctx = hparams.n_audio_ctx;

				const int n_mem = n_text_layer * n_audio_ctx;
				const int n_elements = n_text_state * n_mem;

				model.memory_cross_k = ggml_new_tensor_1d( ctx, GGML_TYPE_F16, n_elements );
				model.memory_cross_v = ggml_new_tensor_1d( ctx, GGML_TYPE_F16, n_elements );
			}

			const size_t memory_size =
				ggml_nbytes( model.memory_k ) + ggml_nbytes( model.memory_v ) +
				ggml_nbytes( model.memory_cross_k ) + ggml_nbytes( model.memory_cross_v );

			logDebug( u8"%s: memory size   = %7.2f MB", __func__, memory_size / 1024.0 / 1024.0 );
		}

		// load weights
		{
			size_t total_size = 0;
			int n_loaded = 0;
			std::string name;

			while( true )
			{
				int32_t n_dims;
				int32_t length;
				int32_t ftype;

				HRESULT hr = readStruct( stm, n_dims );
				if( hr == E_EOF )
					break;
				CHECK( hr );
				CHECK( readStruct( stm, length ) );
				CHECK( readStruct( stm, ftype ) );

				int32_t nelements = 1;
				int32_t ne[ 3 ] = { 1, 1, 1 };
				for( int i = 0; i < n_dims; ++i )
				{
					CHECK( readStruct( stm, ne[ i ] ) );
					nelements *= ne[ i ];
				}

				name.resize( length );
				CHECK( readBytes( stm, name.data(), length ) );

				if( tensors.find( name.data() ) == tensors.end() )
				{
					logError( u8"%s: unknown tensor '%s' in model file", __func__, name.data() );
					return E_INVALIDARG;
				}

				auto tensor = tensors[ name.data() ];
				if( ggml_nelements( tensor ) != nelements )
				{
					logError( u8"%s: tensor '%s' has wrong size in model file", __func__, name.data() );
					return E_INVALIDARG;
				}

				if( tensor->ne[ 0 ] != ne[ 0 ] || tensor->ne[ 1 ] != ne[ 1 ] || tensor->ne[ 2 ] != ne[ 2 ] )
				{
					logError( u8"%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]",
						__func__, name.data(), tensor->ne[ 0 ], tensor->ne[ 1 ], tensor->ne[ 2 ], ne[ 0 ], ne[ 1 ], ne[ 2 ] );
					return E_INVALIDARG;
				}

				const size_t bpe = ( ftype == 0 ) ? sizeof( float ) : sizeof( ggml_fp16_t );

				if( nelements * bpe != ggml_nbytes( tensor ) )
				{
					logError( u8"%s: tensor '%s' has wrong size in model file: got %zu, expected %zu",
						__func__, name.data(), ggml_nbytes( tensor ), nelements * bpe );
					return E_INVALIDARG;
				}

				CHECK( readBytes( stm, tensor->data, ggml_nbytes( tensor ) ) );

				//printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
				total_size += ggml_nbytes( tensor );
				n_loaded++;
				// loader.tryLoad( tensor );
			}

			logDebug( u8"%s: model size    = %7.2f MB", __func__, total_size / 1024.0 / 1024.0 );
			if( n_loaded == 0 )
			{
				logError( u8"%s: no tensors loaded from model file", __func__ );
				return E_INVALIDARG;
			}
			else if( n_loaded != (int)tensors.size() )
			{
				logError( u8"%s: not all tensors loaded from model file - expected %zu, got %d", __func__, tensors.size(), n_loaded );
				return E_INVALIDARG;
			}
			model.n_loaded = n_loaded;
		}

		return S_OK;
	}

	HRESULT Context::load( iReadStream* stm )
	{
		const int64_t t_start_us = ggml_time_us();
		ctx.t_start_us = t_start_us;
		HRESULT hr = loadImpl( stm );
		ctx.t_load_us = ggml_time_us() - t_start_us;
		return hr;
	}

	HRESULT __stdcall loadReferenceCpuModel( const wchar_t* path, iModel** pp )
	{
		if( nullptr == path || nullptr == pp )
			return E_POINTER;

		ComLight::Object<ReadStream> stream;
		CHECK( stream.open( path ) );

		ggml_time_init();
		ComLight::CComPtr<ComLight::Object<Context>> obj;
		CHECK( ComLight::Object<Context>::create( obj ) );
		CHECK( obj->load( &stream ) );
		obj.detach( pp );
		return S_OK;
	}
}

#include "Whisper/WhisperContext.h"
#include "Whisper/ModelBuffers.h"
#include "ML/testUtils.h"
using namespace DirectCompute;

static DirectCompute::Tensor gpuEncode( const whisper_context& wctx, const int mel_offset )
{
	return DirectCompute::Tensor{};
#if 0
	using namespace DirectCompute;
	WhisperContext& ctx = WhisperContext::current();

	Tensor cur;
	sEncodeParams whisperParams;
	const auto& mel_inp = wctx.mel;
	{
		const auto& model = wctx.model;
		const auto& hparams = model.hparams;
		whisperParams.n_len = (uint32_t)mel_inp.n_len;
		whisperParams.n_mel = (uint32_t)mel_inp.n_mel;

		const int n_ctx = wctx.exp_n_audio_ctx > 0 ? wctx.exp_n_audio_ctx : wctx.model.hparams.n_audio_ctx;
		assert( n_ctx > 0 );
		whisperParams.n_ctx = (uint32_t)n_ctx;

		const int n_mels = hparams.n_mels;
		assert( n_mels > 0 );
		whisperParams.n_mels = (uint32_t)n_mels;

		assert( mel_offset >= 0 );
		whisperParams.mel_offset = (uint32_t)mel_offset;

		const int layersCount = hparams.n_audio_layer;
		assert( layersCount > 0 );
		whisperParams.layersCount = (uint32_t)layersCount;

		const int n_state = hparams.n_audio_state;
		const int n_head = hparams.n_audio_head;
		assert( n_state >= 0 );
		assert( n_head >= 0 );

		whisperParams.n_state = (uint32_t)n_state;
		whisperParams.n_head = (uint32_t)n_head;

		int n_audio_ctx = hparams.n_audio_ctx;
		assert( n_audio_ctx > 0 );
		whisperParams.n_audio_ctx = (uint32_t)n_audio_ctx;

		int n_text_state = hparams.n_text_state;
		assert( n_text_state > 0 );
		whisperParams.n_text_state = (uint32_t)n_text_state;

		int n_text_layer = hparams.n_text_layer;
		assert( n_text_layer > 0 );
		whisperParams.n_text_layer = (uint32_t)n_text_layer;

		int n_text_ctx = hparams.n_text_ctx;
		assert( n_text_ctx > 0 );
		whisperParams.n_text_ctx = (uint32_t)n_text_ctx;
	}

	return ctx.encode( mel_inp.data, whisperParams );
#endif
}

GpuEncTest::GpuEncTest( const whisper_context& wctx, const int mel_offset )
{
	return;
	gpuResult = gpuEncode( wctx, mel_offset );
}

void GpuEncTest::compare( const ggml_tensor* expected ) const
{
	return;
	WhisperContext& ctx = WhisperContext::current();
	ctx.dbgPrintDifference( expected, gpuResult, "GpuEncTest.compare", false );
}

void GpuEncTest::compareMel( const ggml_tensor* expected ) const
{
	return;
	WhisperContext& ctx = WhisperContext::current();
	ctx.dbgPrintDifference( expected, mel, "GpuEncTest.compareMel", false );
}

/*
void GpuEncTest::comparePostponed()
{
	if( nullptr == tempRef )
		return;

	WhisperContext& ctx = WhisperContext::current();
	ctx.dbgPrintDifference( tempRef, tempGpu, "comparePostponed" );
	tempRef = nullptr;
} */

__declspec( noinline ) GpuDecTest::GpuDecTest( const whisper_context& wctx, const int* tokens, const int n_tokens, const int n_past )
{
#if 1
	return;
#else
	sDecodeParams dp;
	{
		WhisperContext& ctx = WhisperContext::current();
		const auto& model = wctx.model;
		const auto& hparams = model.hparams;
		dp.n_state = hparams.n_text_state;
		dp.n_head = hparams.n_text_head;
		dp.n_ctx = hparams.n_text_ctx;
		dp.n_past = n_past;
		dp.M = wctx.exp_n_audio_ctx > 0 ? wctx.exp_n_audio_ctx : hparams.n_audio_ctx;
		dp.n_text_layer = hparams.n_text_layer;
		dp.n_vocab = hparams.n_vocab;
	}

	WhisperContext& ctx = WhisperContext::current();
	ctx.decode( tokens, n_tokens, dp, logits, probs );
#endif
}

void __declspec( noinline ) GpuDecTest::compare( const std::vector<float>& cpuLogits, const std::vector<float>& cpuProbs ) const
{
	return;

	if( cpuLogits.size() != logits.size() )
	{
		printf( "GpuDecTest.compare fail, size different\n" );
		return;
	}

	computeDiff( logits.data(), cpuLogits.data(), logits.size() ).print( "GpuDecTest.compare logits" );
	computeDiff( probs.data(), cpuProbs.data(), probs.size() ).print( "GpuDecTest.compare probs" );
}

void __declspec( noinline ) GpuDecTest::postpone( const ggml_tensor* t )
{
	return;

	if( nullptr != tempRef )
		return;
	tempRef = t;
}

void __declspec( noinline ) GpuDecTest::comparePostponed()
{
#if 1
	return;
#else
	if( nullptr == tempRef )
		return;
	WhisperContext& ctx = WhisperContext::current();
	ID3D11ShaderResourceView* srv = ctx.dbgDecodeTest;
	if( nullptr == srv )
		return;

	ctx.dbgPrintDifference( tempRef, ctx.dbgDecodeTest, "GpuDecTest.comparePostponed" );
	tempRef = nullptr;
#endif
}
#else
HRESULT __stdcall Whisper::loadReferenceCpuModel( const wchar_t* path, Whisper::iModel** pp )
{
	logError( u8"This build of the DLL doesn’t implement the reference CPU-running Whisper model." );
	return E_NOTIMPL;
}
#endif