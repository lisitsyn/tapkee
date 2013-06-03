/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_OPENCL_H_
#define TAPKEE_OPENCL_H_

#include <CL/cl.hpp>
#include <iostream>

namespace tapkee
{
namespace tapkee_internal
{

class OpenCL
{
	private:
		cl_int error_code;
		std::vector<cl::Device> devices_;
		std::vector<cl::Platform> platforms_;
		cl::Context context_;
		cl::CommandQueue queue_;
		cl::Program program_;
		cl::Program::Sources sources_;

		OpenCL() : error_code(CL_SUCCESS)
		{
		}

	public:

		static OpenCL& instance() 
		{
			static OpenCL ocl;
			return ocl;
		}

		void addSource(const char* source)
		{
			sources_.push_back(std::make_pair(source, strlen(source)));
		}

		template <typename T>
		cl::Buffer readOnlyBuffer(T* ptr, unsigned int size)
		{
			return cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			                  sizeof(T)*size, ptr);
		}

		template <typename T>
		cl::Buffer readWriteBuffer(unsigned int size)
		{
			return cl::Buffer(context_, CL_MEM_READ_WRITE,
			                  sizeof(T)*size);
		}

		void load(const cl::Buffer& buffer, DenseMatrix& matrix)
		{
			queue_.enqueueReadBuffer(buffer, false, 0, sizeof(ScalarType)*matrix.cols()*matrix.rows(), matrix.data());
		}

		cl::KernelFunctor kernel(const char* name, unsigned int x)
		{
			cl::Kernel kernel(program_, name, &error_code);
			return cl::KernelFunctor(kernel, queue_, cl::NullRange,
			                         cl::NDRange(x), cl::NullRange);
		}

		cl::KernelFunctor kernel(const char* name, unsigned int x, unsigned int y)
		{
			cl::Kernel kernel(program_, name, &error_code);
			return cl::KernelFunctor(kernel, queue_, cl::NullRange,
			                         cl::NDRange(x, y), cl::NullRange);
		}
		
		cl::KernelFunctor kernel(const char* name, unsigned int x, unsigned int y, unsigned int z)
		{
			cl::Kernel kernel(program_, name, &error_code);
			return cl::KernelFunctor(kernel, queue_, cl::NullRange,
			                         cl::NDRange(x, y, z), cl::NullRange);
		}

		void buildPrograms()
		{
			error_code = cl::Platform::get(&platforms_);
			
			cl_context_properties properties[] = 
				{CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms_[0])(), 0};
			context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);
			
			devices_ = context_.getInfo<CL_CONTEXT_DEVICES>(&error_code);
			
			program_ = cl::Program(context_, sources_);
			
			error_code = program_.build(devices_);
			if (error_code != CL_SUCCESS)
				std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[0]) << std::endl;

			queue_ = cl::CommandQueue(context_, devices_[0], 0, &error_code);
		}
};

}
}
#endif
