#pragma once
#include "../../ComLightLib/streams.h"
#include "../API/SpecialTokens.h"
#include "../Utils/MurmurHash3.h"

namespace Whisper
{
	class Vocabulary
	{
		std::vector<const char*> tokens;
		std::vector<char> stringData;
		using THashMap = CAtlMap<const char*, int, StringPtrTraits>;
		THashMap idFromToken;

		void addExtra( int index, const char* format, int i );

		void completeBuild();
	public:
		Vocabulary();

		int n_vocab = 51864;

		HRESULT load( ComLight::iReadStream* stm, int lengthInHeader );

		using id = int;

		id token_eot = 50256;
		id token_sot = 50257;
		id token_prev = 50360;
		id token_solm = 50361; // ??
		id token_not = 50362; // no timestamps
		id token_beg = 50363;

		// available tasks
		static const id token_translate = 50358;
		static const id token_transcribe = 50359;

		bool is_multilingual() const
		{
			return n_vocab == 51865;
		}

		const char* string( int id ) const
		{
			if( id >= 0 && id < (int)tokens.size() )
				return tokens[ id ];
			return nullptr;
		}

		int findId( const char* token ) const;
		int findId( const std::string& token ) const
		{
			return findId( token.c_str() );
		}

		size_t size() const
		{
			return tokens.size();
		}

		void getSpecialTokens( SpecialTokens& rdi ) const;

		size_t getMemoryUse() const
		{
			return vectorMemoryUse( tokens ) + vectorMemoryUse( stringData );
		}

		HRESULT tokenize( const std::string& text, std::vector<id>& tokens ) const;
	};
}