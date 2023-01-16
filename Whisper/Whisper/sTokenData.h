#pragma once
#include <stdint.h>

namespace Whisper
{
	using whisper_token = int;

	struct sTokenData
	{
		whisper_token id;  // token id
		whisper_token tid; // forced timestamp token id
		float p;           // probability of the token
		float pt;          // probability of the timestamp token
		float ptsum;       // sum of probabilities of all timestamp tokens
		float vlen;        // voice length of the token
		// token-level timestamp data
		// do not use if you haven't computed token-level timestamps
		int64_t t0;        // start time of the token
		int64_t t1;        //   end time of the token
	};


}