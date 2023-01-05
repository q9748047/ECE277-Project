# - Try to find the GLFW library
# Once done this will define
#
#  FREEGLUT_FOUND - system has GLFW
#  FREEGLUT_INCLUDE_DIR - the GLFW include directory
#  FREEGLUT_LIBRARIES - The libraries needed to use GLFW
set(GRAPHICS_LIB ${CUDA_TOOLKIT_SAMPLES_DIR}/common)

if(FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARIES)
   set(FREEGLUT_FOUND TRUE)
else(FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARIES)

FIND_PATH(FREEGLUT_INCLUDE_DIR GL/freeglut.h
	${GRAPHICS_LIB}/include
	${GRAPHICS_LIB}/inc
   /usr/include
   /usr/local/include
   $ENV{GLFWROOT}/include
   $ENV{FREEGLUT_ROOT}/include
   $ENV{FREEGLUT_DIR}/include
   $ENV{FREEGLUT_DIR}/inc
   [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.1\\Setup\\VC]/PlatformSDK/Include
)

FIND_LIBRARY(FREEGLUT_DEBUG_LIBRARIES NAMES freeglut
   PATHS
	 ${GRAPHICS_LIB}/lib/x64
   /usr/lib
   /usr/lib64
   /usr/local/lib
   /usr/local/lib64
   $ENV{GLFWROOT}/lib
   $ENV{FREEGLUT_ROOT}/lib
   $ENV{FREEGLUT_DIR}/lib
   [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.1\\Setup\\VC]/PlatformSDK/Lib
   DOC "FREEGLUT library name"
)

FIND_LIBRARY(FREEGLUT_RELEASE_LIBRARIES NAMES freeglut
   PATHS
	 ${GRAPHICS_LIB}/lib/x64
   /usr/lib
   /usr/lib64
   /usr/local/lib
   /usr/local/lib64
   $ENV{GLFWROOT}/lib
   $ENV{FREEGLUT_ROOT}/lib
   $ENV{FREEGLUT_DIR}/lib
   [HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\9.1\\Setup\\VC]/PlatformSDK/Lib
   DOC "FREEGLUT library name"
)

if(FREEGLUT_INCLUDE_DIR AND FREEGLUT_DEBUG_LIBRARIES AND FREEGLUT_RELEASE_LIBRARIES)
   set(FREEGLUT_FOUND TRUE)
endif(FREEGLUT_INCLUDE_DIR AND FREEGLUT_DEBUG_LIBRARIES AND FREEGLUT_RELEASE_LIBRARIES)


if(FREEGLUT_FOUND)
   if(NOT FREEGLUT_FIND_QUIETLY)
      message(STATUS "Found FREEGLUT: ${FREEGLUT_DEBUG_LIBRARIES}")
   endif(NOT FREEGLUT_FIND_QUIETLY)
else(FREEGLUT_FOUND)
   if(FREEGLUT_FIND_REQUIRED)
      message(FATAL_ERROR "could NOT find FREEGLUT")
   endif(FREEGLUT_FIND_REQUIRED)
endif(FREEGLUT_FOUND)

MARK_AS_ADVANCED(FREEGLUT_INCLUDE_DIR FREEGLUT_LIBRARIES)

endif(FREEGLUT_INCLUDE_DIR AND FREEGLUT_LIBRARIES)
