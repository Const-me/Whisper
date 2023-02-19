#pragma once
#include "Tensor.h"

namespace DirectCompute
{
	using pfnNewCapacity = uint32_t( * )( uint32_t current, uint32_t requested );

	uint32_t defaultNewCapacity( uint32_t current, uint32_t requested );

	class PooledTensor
	{
		TensorGpuViews views;
		uint32_t capacity = 0;
	public:
		Tensor tensor( eDataType type, const std::array<uint32_t, 4>& ne, pfnNewCapacity pfnNewCap );
		size_t getCapacity() const { return capacity; }
		void clear()
		{
			views.clear();
			capacity = 0;
		}
		HRESULT zeroMemory();
	};

	__interface iTensorArena
	{
		Tensor tensor( eDataType type, const std::array<uint32_t, 4>& ne );
		void reset();
	};

	class TensorsArena: public iTensorArena
	{
	public:
		struct sArenaConfig
		{
			pfnNewCapacity pfnCapInner;
			size_t initialCapOuter;
		};

		struct sArenaConfigs
		{
			sArenaConfig fp16, fp32;
		};

		TensorsArena( const sArenaConfigs& configs );

		Tensor tensor( eDataType type, const std::array<uint32_t, 4>& ne ) override final;
		void reset() override final;

		void clear();
		__m128i getMemoryUse() const;
		HRESULT zeroMemory();

	private:

		struct ArenaImpl
		{
			ArenaImpl( eDataType dataType, const sArenaConfig& config );

			void reset()
			{
				index = 0;
			}

			void clear()
			{
				index = 0;
				pool.clear();
			}

			Tensor tensor( const std::array<uint32_t, 4>& ne );
			__m128i getMemoryUse() const;
			HRESULT zeroMemory();

		private:

			const eDataType type;
			const pfnNewCapacity pfnNewCap;

			std::vector<PooledTensor> pool;
			size_t index = 0;
		};

		static constexpr size_t countTypes = 2;
		std::array<ArenaImpl, countTypes> arenas;
	};
}