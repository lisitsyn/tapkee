#include <Python.h>

#include "tapkee_module.hpp"

extern "C" {

// Function to initialize the pytapkee module
PyMODINIT_FUNC PyInit_libpytapkee(void) {
    // Initialize the module
    PyObject* module = PyModule_Create(&pytapkee_module);

    // If initialization failed
    if (module == NULL) {
        return NULL;
    }

    // You can add module-level attributes here if needed
    // e.g., PyModule_AddStringConstant(module, "__author__", "Your Name");

    return module;
}

}  // extern "C"

