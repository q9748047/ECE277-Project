1. Use CMake to generate the Visual Studio 2019 solution with source code in folder ''C++''.
2. Open the .sln file with Visual Studio 2019.
3. Build each project (cu_matrix_add, cu_matrix_sub, cu_matrix_mul, cu_matrix_trans) in Release mode.
4. Copy the generated files (.pyd, .lib, .exp) to folder ''python'' and import them as modules in Python.

In the folder ''python'', I have put all generated files and wrote ''matrix.py'' to assemble them.
''test_add.py'', ''test_sub.py'', ''test_mul.py'', ''test_trans.py'' are codes to test each operation type. They can be executed directly with Python.