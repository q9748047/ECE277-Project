#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_mtrans(int* A, int* B, int M, int N);

namespace py = pybind11;


py::array_t<int> mtrans_wrapper(py::array_t<int> a1) 
{
	auto buf1 = a1.request();

	if (a1.ndim() != 2)
		throw std::runtime_error("Number of dimensions must be two");

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];
	printf("M=%d, N=%d\n", M, N);

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(int),     /* Size of one item */
		py::format_descriptor<int>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ M, N},  /* Number of elements for each dimension */
		{ sizeof(int)*N, sizeof(int) }  /* Strides for each dimension */
	));

	auto buf2 = result.request();

	int* A = (int*)buf1.ptr;
	int* B = (int*)buf2.ptr;

	cu_mtrans(A, B, M, N);

    return result;
}



PYBIND11_MODULE(cu_matrix_trans, m) {
    m.def("mtrans", &mtrans_wrapper, "Transcope a NumPy array");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
