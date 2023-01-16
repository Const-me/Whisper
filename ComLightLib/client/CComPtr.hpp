#pragma once

namespace ComLight
{
	// COM smart pointer, very comparable to CComPtr from ATL
	template <class I>
	class CComPtr
	{
		I* p;

		void callAddRef() const
		{
			if( nullptr == p )
				return;
			p->AddRef();
		}

	public:

		// Construct with nullptr
		CComPtr() : p( nullptr ) { }

		// Release the pointer
		void release()
		{
			if( nullptr == p )
				return;
			p->Release();
			p = nullptr;
		}

		~CComPtr()
		{
			release();
		}

		// Attach without AddRef()
		void attach( I* raw )
		{
			release();
			p = raw;
		}

		// Detach without Release(), set this pointer to nullptr
		I* detach()
		{
			I* const result = p;
			p = nullptr;
			return result;
		}

		// Detach without Release() and place to the specified address, set this pointer to nullptr
		template<class Other>
		void detach( Other** pp )
		{
			// If the argument points to a non-empty object, release the old instance: would leak memory otherwise.
			if( nullptr != *pp )
				( *pp )->Release();
			( *pp ) = detach();
		}

		// Set and AddRef()
		void assign( I* raw )
		{
			release();
			attach( raw );
			callAddRef();
		}

		void swap( CComPtr<I>& that )
		{
			std::swap( p, that.p );
		}

		// Set and AddRef()
		CComPtr( I* raw ) : p( raw )
		{
			callAddRef();
		}

		// Set and AddRef()
		CComPtr( const CComPtr<I>& that ) : CComPtr( that.p ) { }
		// Move constructor
		CComPtr( CComPtr<I>&& that ) : p( that.p ) { that.p = nullptr; }

		// Set and AddRef()
		void operator=( I* raw )
		{
			assign( raw );
		}

		// Set and AddRef()
		void operator=( const CComPtr<I>& that )
		{
			assign( that.p );
		}

		// Move assignment operator, destroys the other one
		void operator=( CComPtr<I>&& that )
		{
			attach( that.detach() );
		}

		operator I*( ) const { return p; }
		I* operator -> () const { return p; }
		I** operator &() { return &p; }

		operator bool() const { return nullptr != p; }
	};
}