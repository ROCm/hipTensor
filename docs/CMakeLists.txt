find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

add_custom_target(copyover_install
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../INSTALL.md ${CMAKE_CURRENT_SOURCE_DIR}/install.md
    COMMENT "Copying over INSTALL.md to docs folder as install.md."
)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/include)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                     MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                     COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE} copyover_install)

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_HOST ${CMAKE_CURRENT_SOURCE_DIR}/_build)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)

add_custom_target(Sphinx ALL
                  COMMAND ${SPHINX_EXECUTABLE} -b html
                  # Tell Breathe where to find the Doxygen output
                  -Dbreathe_projects.proj=${DOXYGEN_OUTPUT_DIR}/xml
                  -Dbreathe_default_project=proj
		  ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating HTML documentation with Sphinx")
	 
add_custom_target(Sphinx-host ALL
                  COMMAND ${SPHINX_EXECUTABLE} -b html
                  # Tell Breathe where to find the Doxygen output
                  -Dbreathe_projects.proj=${DOXYGEN_OUTPUT_DIR}/xml
                  -Dbreathe_default_project=proj
		  ${SPHINX_SOURCE} ${SPHINX_HOST}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Generating HTML documentation to host")

set(SPHINX_LATEX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx/latex)
add_custom_target(Sphinx-latex ALL
                   COMMAND ${SPHINX_EXECUTABLE} -b latex
                   # Tell Breathe where to find the Doxygen output
                   -Dbreathe_projects.proj=${DOXYGEN_OUTPUT_DIR}/xml
                   -Dbreathe_default_project=proj
                   ${SPHINX_SOURCE} ${SPHINX_LATEX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   COMMENT "Generating LATEX documentation with Sphinx"
                  )
add_custom_target(Sphinx-build_pdf ALL
                   COMMAND make 
                   WORKING_DIRECTORY ${SPHINX_LATEX_BUILD}
                   COMMENT "Building PDF on the LATEX documentation built with Sphinx")
