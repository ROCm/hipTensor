#Look for an executable called sphinx-build
find_program(SPHINX_EXECUTABLE 
    NAMES sphinx-build
    DOC "Sphinx documentation generator"
)
mark_as_advanced(SPHINX_EXECUTABLE)

#TOFIX the pointing to /usr/bin instead of /usr/local/bin issue
set(SPHINX_EXECUTABLE "/usr/local/bin/sphinx-build")

include(FindPackageHandleStandardArgs)

#Handle standard arguments to find_package like REQUIRED and QUIET
find_package_handle_standard_args(Sphinx
                                  "Failed to find sphinx-build executable"
                                  SPHINX_EXECUTABLE)
