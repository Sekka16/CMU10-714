#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  size_t cur_gid = gid;
  size_t idx = offset;
  for (int i = shape.size - 1; i >= 0; --i) {
    size_t dim_size = shape.data[i];
    size_t coord = cur_gid % dim_size;
    idx += coord * strides.data[i];
    cur_gid /= dim_size;
  }
  if (gid < size)  
    out[gid] = a[idx];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, 
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  size_t cur_gid = gid;
  size_t idx = offset;
  for (int i = shape.size - 1; i >= 0; --i) {
    size_t dim_size = shape.data[i];
    size_t coord = cur_gid % dim_size;
    idx += coord * strides.data[i];
    cur_gid /= dim_size;
  }
  if (gid < size)  
    out[idx] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape), 
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec shape, 
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t cur_gid = gid;
  size_t idx = offset;
  for (int i = shape.size - 1; i >= 0; --i) {
    size_t dim_size = shape.data[i];
    size_t coord = cur_gid % dim_size;
    idx += coord * strides.data[i];
    cur_gid /= dim_size;
  }
  if (gid < size)  
    out[idx] = val;  
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
template <typename BinaryOp>
__global__ void EwiseBinaryKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, BinaryOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], b[gid]);
}

template <typename BinaryOp>
__global__ void ScalarBinaryKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, BinaryOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], val);
}

template <typename UnaryOp>
__global__ void EwiseUnaryKernel(const scalar_t* a, scalar_t* out, size_t size, UnaryOp op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid]);
}

struct AddOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x + y; }
  static __device__ constexpr scalar_t identity() { return 0.0f; }
};

struct MulOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x * y; }
};

struct DivOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x / y; }
};

struct PowerOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return powf(x, y); }
};

struct MaximumOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return max(x, y); }
  static __device__ constexpr scalar_t identity() { return 0xFF800000; }  
};

struct EqOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return (x == y) ? 1.0f : 0.0f; }
};

struct GeOp {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return (x >= y) ? 1.0f : 0.0f; }
};

struct LogOp {
  __device__ scalar_t operator()(scalar_t x) const { return logf(x); }
};

struct ExpOp {
  __device__ scalar_t operator()(scalar_t x) const { return expf(x); }
};

struct TanhOp {
  __device__ scalar_t operator()(scalar_t x) const { return tanhf(x); }
};

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, AddOp{});
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, AddOp{});
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out)  {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, DivOp{});
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, DivOp{});
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MulOp{});
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MulOp{});
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, PowerOp{});
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MaximumOp{});
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MaximumOp{});
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EqOp{});
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, EqOp{});
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, GeOp{});
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarBinaryKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, GeOp{});
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, LogOp{});
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, ExpOp{});
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseUnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, TanhOp{});
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Matmul operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(scalar_t* a, scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
  return;
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
constexpr size_t kWarpSize = 32;

template<typename Op>
__device__ scalar_t WarpReduce(scalar_t val, Op op) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = op(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

template<typename Op>
__device__ scalar_t BlockReduce(scalar_t val, Op op) {
  const int NUM_WARPS = (BASE_THREAD_NUM + kWarpSize - 1) / kWarpSize;
  __shared__ scalar_t shared[NUM_WARPS];
  __shared__ scalar_t block_result;

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;

  val = WarpReduce(val, op);
  if (lane == 0) shared[warp] = val;
  __syncthreads();

  if (warp == 0) {
    scalar_t result = (lane < NUM_WARPS) ? shared[lane] : Op::identity();
    val = WarpReduce(result, op);
    if (lane == 0) { block_result = val; }
  }
  return block_result;
}

template<typename Op>
__global__ void WarpReduceKernel(const scalar_t* a, scalar_t* out, size_t m, size_t n, Op op) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int warp_nums = blockDim.x / kWarpSize;

  for (int row_start = blockIdx.x * warp_nums; row_start < m; row_start += gridDim.x * warp_nums) {
    const int row = row_start + warp;
    if (row < m) {
      scalar_t res = Op::identity();
      for (int col_start = lane; col_start < n; col_start += kWarpSize) {
        scalar_t val = a[row * n + col_start];
        res = op(res, val);
      } 
      res = WarpReduce(res, op);
      if (lane == 0) out[row] = res;
    }
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim;
  const int rows_per_block = BASE_THREAD_NUM / kWarpSize;
  dim.grid = (out->size + rows_per_block - 1) / rows_per_block;
  dim.block = BASE_THREAD_NUM;
  WarpReduceKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size, MaximumOp{});
  /// END SOLUTION
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim;
  const int rows_per_block = BASE_THREAD_NUM / kWarpSize;
  dim.grid = (out->size + rows_per_block - 1) / rows_per_block;
  dim.block = BASE_THREAD_NUM;
  WarpReduceKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size, AddOp{});
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}