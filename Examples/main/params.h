#pragma once
#include <vector>
#include <string>

// command-line parameters
struct whisper_params
{
	uint32_t n_threads;
	uint32_t n_processors = 1;
	uint32_t offset_t_ms = 0;
	uint32_t offset_n = 0;
	uint32_t duration_ms = 0;
	uint32_t max_context = UINT_MAX;
	uint32_t max_len = 0;

	float word_thold = 0.01f;

	bool speed_up = false;
	bool translate = false;
	bool diarize = false;
	bool output_txt = false;
	bool output_vtt = false;
	bool output_srt = false;
	bool output_wts = false;
	bool print_special = false;
	bool print_colors = true;
	bool no_timestamps = false;

	std::string language = "en";
	std::wstring model = L"models/ggml-base.en.bin";
	std::wstring gpu;
	std::string prompt;
	std::vector<std::wstring> fname_inp;

	whisper_params();

	bool parse( int argc, wchar_t* argv[] );
};

void whisper_print_usage( int argc, wchar_t** argv, const whisper_params& params );