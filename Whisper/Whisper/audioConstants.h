#pragma once
#include <stdint.h>

namespace Whisper
{
	// WHISPER_SAMPLE_RATE, 16 kHz
	constexpr uint32_t SAMPLE_RATE = 16000;
	// WHISPER_N_FFT, 25 milliseconds
	constexpr uint32_t FFT_SIZE = 400;
	// WHISPER_HOP_LENGTH, 10 milliseconds
	constexpr uint32_t FFT_STEP = 160;
	// WHISPER_N_MEL
	constexpr uint32_t N_MEL = 80;
}