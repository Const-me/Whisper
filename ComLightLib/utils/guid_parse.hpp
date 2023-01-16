// https://github.com/tobias-loew/constexpr-GUID-cpp-11

//-------------------------------------------------------------------------------------------------------
// constexpr GUID parsing
// Written by Alexander Bessonov
// Written by Tobias Loew
//
// Licensed under the MIT license.
//-------------------------------------------------------------------------------------------------------

#pragma once
#include <stdexcept>
#include <string>
#include <cassert>
#include <cstdint>

#if !defined(GUID_DEFINED)
#define GUID_DEFINED
struct GUID {
	uint32_t Data1;
	uint16_t Data2;
	uint16_t Data3;
	uint8_t Data4[ 8 ];
};
#endif

namespace ComLight
{
	namespace details
	{
		constexpr const size_t short_guid_form_length = 36;	// XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
		constexpr const size_t long_guid_form_length = 38;	// {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}

		constexpr uint8_t parse_hex_digit( const char c )
		{
			using namespace std::string_literals;
			return
				( '0' <= c && c <= '9' )
				? c - '0'
				: ( 'a' <= c && c <= 'f' )
				? 10 + c - 'a'
				: ( 'A' <= c && c <= 'F' )
				? 10 + c - 'A'
				:
				throw std::domain_error{ "invalid character in GUID"s };
		}

		constexpr uint8_t parse_hex_uint8_t( const char *ptr )
		{
			return ( parse_hex_digit( ptr[ 0 ] ) << 4 ) + parse_hex_digit( ptr[ 1 ] );
		}

		constexpr uint16_t parse_hex_uint16_t( const char *ptr )
		{
			return ( parse_hex_uint8_t( ptr ) << 8 ) + parse_hex_uint8_t( ptr + 2 );
		}

		constexpr uint32_t parse_hex_uint32_t( const char *ptr )
		{
			return ( parse_hex_uint16_t( ptr ) << 16 ) + parse_hex_uint16_t( ptr + 4 );
		}

		constexpr GUID parse_guid( const char *begin )
		{
			return GUID{
				parse_hex_uint32_t( begin ),
				parse_hex_uint16_t( begin + 8 + 1 ),
				parse_hex_uint16_t( begin + 8 + 1 + 4 + 1 ),
				{
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 + 2 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 + 2 + 2 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 + 2 + 2 + 2 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 + 2 + 2 + 2 + 2 ),
					parse_hex_uint8_t( begin + 8 + 1 + 4 + 1 + 4 + 1 + 2 + 2 + 1 + 2 + 2 + 2 + 2 + 2 )
				}
			};
		}

		constexpr GUID make_guid_helper( const char *str, size_t N )
		{
			using namespace std::string_literals;
			using namespace details;

			return ( !( N == long_guid_form_length || N == short_guid_form_length ) )
				? throw std::domain_error{ "String GUID of the form {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} or XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX is expected"s }
				: ( N == long_guid_form_length && ( str[ 0 ] != '{' || str[ long_guid_form_length - 1 ] != '}' ) )
				? throw std::domain_error{ "Missing opening or closing brace"s }

			: parse_guid( str + ( N == long_guid_form_length ? 1 : 0 ) );
		}


		template<size_t N>
		constexpr GUID make_guid( const char( &str )[ N ] )
		{
			return make_guid_helper( str, N - 1 );
		}
	}
	using details::make_guid;
}