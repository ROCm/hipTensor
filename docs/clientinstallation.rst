Setting up the client application
=================================

| Assuming hiptensor has be installed to the path specified in the ${CMAKE_INSTALL_PREFIX} while building the hiptensor package.
| Update the environment variables as mentioned below.

.. code-block:: console

   export hiptensor_ROOT=${CMAKE_INSTALL_PREFIX}
   export LD_LIBRARY_PATH=${hiptensor_ROOT}/lib/:${LD_LIBRARY_PATH}

| If we store the following code in a file called contraction.cpp, we can compile it via the following command:

.. code-block:: console

   hipcc contraction.cpp -L${hiptensor_ROOT}/lib/ -I${hiptensor_ROOT}/include -std=c++17 -lhiptensor -o contraction

| When compiling intermediate steps of this example, the compiler might warn about unused variables. This is due to the example not being complete. The final step should issue no warnings.

| Run the executable as mentioned below and it should return the results without any warnings or errors.

.. code-block:: console

   ./contraction

