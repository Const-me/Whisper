#pragma once
#include "../D3D/enums.h"
#include "../ML/TensorShape.h"
// 1 = new tensors can be allocated with `nullptr` iMemoryAllocator, by allocating memory internally and counting these references
// 0 = memory allocator is mandatory, create() methods will fail with E_POINTER if the allocator is `nullptr`
#define TENSOR_INTERNAL_ALLOC 0

// 1 = expose compatibility API for GGML interop
#define TENSOR_GGML_COMPAT 0

#if TENSOR_GGML_COMPAT
#include "../source/ggml.h"
#endif

namespace CpuCompute
{
	using DirectCompute::TensorShape;
	using DirectCompute::eDataType;

	__interface iMemoryAllocator
	{
		void* allocate( size_t cb, size_t align );
	};
	__interface iArenaAllocator : public iMemoryAllocator
	{
		void resetArena();
	};

#if TENSOR_GGML_COMPAT
	class Tensor;
	class GgmlTensorView
	{
		ggml_tensor tensor;
	public:
		GgmlTensorView( const Tensor& t );
		operator ggml_tensor* ( ) { return &tensor; }
	};
#endif

	// A functional equivalent of ggml_tensor structure, designed for use from C++
	class Tensor : public TensorShape
	{
		void* m_data = nullptr;

		eDataType m_type = (eDataType)0xFF;

#if TENSOR_INTERNAL_ALLOC
		// True when the memory block was allocated internally by this class
		// In this case, this class does reference counting to support cheap copies.
		// False when it's owned by someone else, such as iMemoryAllocator object, or a GGML's tensor
		bool ownsMemory = false;
		void deallocate();
#endif

		// Private constructors for fromData() methods
		Tensor( void* pointer, eDataType type, std::initializer_list<uint32_t> size );
		Tensor( void* pointer, eDataType type, uint32_t length ) noexcept;
	public:
		// Trivial constructors
		Tensor() = default;
#if TENSOR_INTERNAL_ALLOC
		~Tensor()
		{
			deallocate();
		}
#else
		~Tensor() = default;
#endif
		Tensor( Tensor&& that ) noexcept;
		void operator=( Tensor&& that ) noexcept;
		Tensor( const Tensor& that );
		void operator=( const Tensor& that );

		// Allocate a new tensor
		HRESULT create( eDataType type, const std::array<uint32_t, 4>& sizeElements, iMemoryAllocator* alloc = nullptr );
		// Allocate a new tensor
		HRESULT create( eDataType type, std::initializer_list<uint32_t> sizeElements, iMemoryAllocator* alloc = nullptr );
		// Attach to pre-existing block of memory, interpreting the data as a dense tensor of the specified type and size
		HRESULT attach( void* pointer, eDataType type, std::initializer_list<uint32_t> sizeElements );
		// Attach to pre-existing block of memory, interpret the data as a dense vector of the specified type and length
		static Tensor fromData( void* pointer, eDataType type, uint32_t length );

		eDataType type() const { return m_type; }
		void* data() const { return m_data; }

		uint16_t* fp16()
		{
			assert( m_type == eDataType::FP16 );
			assert( nullptr != m_data );
			return (uint16_t*)m_data;
		}
		const uint16_t* fp16() const
		{
			assert( m_type == eDataType::FP16 );
			assert( nullptr != m_data );
			return (uint16_t*)m_data;
		}
		float* fp32()
		{
			assert( m_type == eDataType::FP32 );
			assert( nullptr != m_data );
			return (float*)m_data;
		}
		const float* fp32() const
		{
			assert( m_type == eDataType::FP32 );
			assert( nullptr != m_data );
			return (float*)m_data;
		}

		Tensor reshape3d( uint32_t ne0, uint32_t ne1, uint32_t ne2 ) const;

		void setType( eDataType dt )
		{
			m_type = dt;
		}
		void setDataPointer( void* pv )
		{
			m_data = pv;
		}

#if TENSOR_GGML_COMPAT
		// Compatibility with GGML's tensors, for testing and lulz
		Tensor( const ggml_tensor* ggml );
		ggml_tensor ggml() const;

		operator GgmlTensorView() const
		{
			return GgmlTensorView( *this );
		}
#endif
	};

	// A pair of tensors containing weights and biases; apparently, both tensors are of the same shape
	struct TensorPair
	{
		Tensor w, b;
	};
}