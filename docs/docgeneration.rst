DOC Generation
==============

| This section describes the generation of documentation for the hipTENSOR package with the Doxygen, Sphinx, and breathe extensions.
  
Pre-requisites
--------------
| All the pre-requistes need to be installed in the docker before generating the documentation for the hipTENSOR package.

* Install Doxygen modules.

.. code-block:: console
  
   sudo apt-get install doxygen doxygen-doc


* Install Latex dependencies required for the doucmentation.

.. code-block:: console
  
   sudo apt-get install texlive-latex-extra
   sudo apt-get install latexmk

| It is recommended to create a virtual environment for the sphinx and other dependencies. The installation steps are mentioned below.

* Install the python-3.9 virtual environment and activate the environment

.. code-block:: console
  
   sudo apt-get install python3.9-venv
 
.. warning:: 
   In a few linux environments, a few errors encounterd. Follow the steps mentioned in the `package errors <https://askubuntu.com/questions/1402410/sub-process-usr-bin-dpkg-returned-an-error-code-1-while-upgrading-python3-10>`_.

* Create and activate the virtual environment and the install the requirments.txt in the docs folders of the package.

.. code-block:: console

    python3.9 -m venv ${VENV_PATH}
    source ${VENV_PATH}/bin/activate
    pip install -r ${PROJCET_SOURCE_DIR}/requirements.txt


Building the documentation
--------------------------

| Initiate the doucmenation of the package using the flag -D BUILD_DOC = ON in cmake arguments as mentioned below.

.. code-block:: console
   
   # Need to specify target ID, example below is gfx908 and gfx90a
   cmake                                                                 \
   -D BUILD_DEV=OFF                                                      \
   -D CMAKE_BUILD_TYPE=Release                                           \
   -D CMAKE_CXX_FLAGS=" --offload-arch=gfx908 --offload-arch=gfx90a -O3" \
   -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                             \
   -D CMAKE_PREFIX_PATH=/opt/rocm                                        \
   -D CMAKE_INSTALL_PREFIX=${PATH_TO_HT_INSTALL_DIRECTORY}               \
   -D BUILD_DOC=ON                                                       \
   ..

| Build the hipTENSOR packages with the same the make command.

.. code-block:: console

   make


HTML and PDF documentation
--------------------------

| After intiating the steps in the earlier mentioned sections, the final documenation available the following paths.

.. code-block:: console
    
   HTML:        ${CMAKE_BINARY_DIR}/docs/sphinx
   PDF/Latex:   ${CMAKE_BINARY_DIR}/docs/sphinx/latex
