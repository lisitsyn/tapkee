#include "tapkee_module.hpp"

// Define the module structure
PyModuleDef pytapkee_module = {
    PyModuleDef_HEAD_INIT,
    "libpytapkee",   // Module name
    NULL,            // Module docstring (optional)
    -1,              // Module state (global state)
    NULL, NULL, NULL, NULL, NULL
};
