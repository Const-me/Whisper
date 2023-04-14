#include "stdafx.h"
#include "Vocabulary.h"
#include "loaderUtils.h"
#include <regex>
using ComLight::iReadStream;
using namespace Whisper;

Vocabulary::Vocabulary() :
	idFromToken( 17u, 0.75f, 0.25f, 1.5f, 1024 )
{ }

void Vocabulary::addExtra( int index, const char* format, int i )
{
	const int len = std::snprintf( nullptr, 0, format, i );
	const size_t offset = stringData.size();
	stringData.resize( offset + len + 1 );
	char* const rdi = stringData.data() + offset;
	std::snprintf( rdi, len + 1, format, i );
	rdi[ len ] = '\0';
	tokens[ index ] = reinterpret_cast<const char*>( offset );
}

void Vocabulary::completeBuild()
{
	stringData.shrink_to_fit();

	// Replace offsets with char pointers
	const size_t dataLength = stringData.size();
	for( auto& s : tokens )
	{
		// The reason this hack works - on Windows, lower 2GB of address space is reserved to the kernel.
		// That's why the strings from the read only section of this DLL like "[_PREV_]" are guaranteed to have their addresses much larger than the size of the data buffer
		const size_t ri = reinterpret_cast<size_t>( s );
		if( ri < dataLength )
			s = stringData.data() + ri;
	}

	// Build hash map to lookup the tokens
	const size_t tokensCount = tokens.size();
	for( size_t i = 0; i < tokensCount; i++ )
		idFromToken.SetAt( tokens[ i ], (int)i );
	idFromToken.Rehash();

	// Log success message
	int64_t cb = stringData.size();
	cb += tokens.size() * sizeof( void* );

	cb += sizeof( void* ) * idFromToken.GetHashTableSize();
	cb += ( sizeof( THashMap::CPair ) + 16 ) * idFromToken.GetCount();

	constexpr double mulKb = 1.0 / ( 1 << 10 );
	logDebug( u8"Loaded vocabulary, %zu strings, %.1f kb RAM", tokens.size(), mulKb * cb );
}

int Vocabulary::findId( const char* token ) const
{
	auto p = idFromToken.Lookup( token );
	if( nullptr != p )
		return p->m_value;
	else
		return -1;
}

HRESULT Vocabulary::load( ComLight::iReadStream* stm, int lengthInHeader )
{
	if( lengthInHeader <= 0 )
		return E_INVALIDARG;

	tokens.clear();
	stringData.clear();

	int countWords = 0;
	CHECK( readStruct( stm, countWords ) );
	if( countWords <= 0 )
		return E_INVALIDARG;

	const size_t count = (uint32_t)countWords;
	const size_t actualCount = std::max( count, (size_t)lengthInHeader );
	tokens.resize( actualCount );

	for( int i = 0; i < count; i++ )
	{
		int countChars = 0;
		CHECK( readStruct( stm, countChars ) );
		if( countChars < 0 )
		{
			logError( u8"Vocabulary.load failed: string length is negative" );
			return E_INVALIDARG;
		}
		if( countChars == 0 )
		{
			// This happens with `ggml-large.bin` and `ggml-large-v1.bin` models.
			// A bug in the model maybe?
			tokens[ i ] = "";
			continue;
		}
		const size_t len = (size_t)countChars;

		const size_t offset = stringData.size();
		stringData.resize( offset + len + 1 );

		CHECK( readBytes( stm, &stringData[ offset ], len ) );
		*stringData.rbegin() = '\0';

		tokens[ i ] = reinterpret_cast<const char*>( offset );
	}

	n_vocab = lengthInHeader;

	if( is_multilingual() )
	{
		token_eot++;
		token_sot++;
		token_prev++;
		token_solm++;
		token_not++;
		token_beg++;
	};

	if( countWords < lengthInHeader )
	{
		for( int i = countWords; i < lengthInHeader; i++ )
		{
			if( i > token_beg )
				addExtra( i, "[_TT_%i]", i - token_beg );
			else if( i == token_eot )
				tokens[ i ] = "[_EOT_]";
			else if( i == token_sot )
				tokens[ i ] = "[_SOT_]";
			else if( i == token_prev )
				tokens[ i ] = "[_PREV_]";
			else if( i == token_not )
				tokens[ i ] = "[_NOT_]";
			else if( i == token_beg )
				tokens[ i ] = "[_BEG_]";
			else
				addExtra( i, "[_extra_token_%i]", i );
		}
	}

	completeBuild();
	return S_OK;
}

void Vocabulary::getSpecialTokens( SpecialTokens& rdi ) const
{
	rdi.TranscriptionEnd = token_eot;
	rdi.TranscriptionStart = token_sot;
	rdi.PreviousWord = token_prev;
	rdi.SentenceStart = token_solm;
	rdi.Not = token_not;
	rdi.TranscriptionBegin = token_beg;
	rdi.TaskTranslate = token_translate;
	rdi.TaskTranscribe = token_transcribe;
}

// https://github.com/ggerganov/whisper.cpp/blob/v1.2.1/whisper.cpp#L2451
HRESULT Vocabulary::tokenize( const std::string& text, std::vector<id>& tokens ) const
{
	std::vector<std::string> words;

	// first split the text into words
	{
		std::string str = text;
		std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
		std::regex re( pat );
		std::smatch m;

		while( std::regex_search( str, m, re ) )
		{
			for( auto x : m )
				words.push_back( x );
			str = m.suffix();
		}
	}

	// find the longest tokens that form the words:
	tokens.clear();
	for( const auto& word : words )
	{
		if( word.empty() )
			continue;

		int i = 0;
		int n = (int)word.size();
		while( i < n )
		{
			int j = n;
			while( j > i )
			{
				const int it = findId( word.substr( i, j - i ) );
				if( it >= 0 )
				{
					tokens.push_back( it );
					i = j;
					break;
				}
				j--;
			}
			if( i == n )
				break;

			if( j == i )
			{
				const auto sub = word.substr( i, 1 );
				const int it = findId( sub );
				if( it >= 0 )
				{
					tokens.push_back( it );
				}
				else
				{
					logError( u8"Unknown token \"%s\"", sub.c_str() );
					return E_INVALIDARG;
				}
				i++;
			}
		}
	}

	return S_OK;
}