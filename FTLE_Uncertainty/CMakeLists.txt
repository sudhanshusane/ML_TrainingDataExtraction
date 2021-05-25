project(ExtractTrainingData)

cmake_minimum_required(VERSION 2.8.10)

set(CMAKE_C_COMPILER /usr/local/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++)
set(CMAKE_C_FLAGS "-g -fopenmp -Ofast")
set(CMAKE_CXX_STANDARD 14)

SET(VTKm_DIR /home/sci/ssane/projects/uncertainty_vis/VTKM/install/lib/cmake/vtkm-1.5)
find_package(VTKm REQUIRED QUIET)

set(CNPY_DIR "/home/sci/ssane/packages/cnpy/install/include")
set(CNPY_LIB "/home/sci/ssane/packages/cnpy/install/lib/libcnpy.so")

set(VTK_DIR "/home/sci/ssane/packages/VTK-install/lib64/cmake/vtk-8.2")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(ExtractTrainingData main.cxx)

  target_include_directories(ExtractTrainingData PRIVATE ${VTKm_INCLUDE_DIRS} ${CNPY_DIR})
  target_link_libraries(ExtractTrainingData PRIVATE vtkm_cont vtkm_io ${VTKm_LIBRARIES} ${VTK_LIBRARIES} ${CNPY_LIB} -fopenmp -lz)
#  target_compile_definitions(ExtractTrainingData PRIVATE "VTKM_DEVICE_ADAPTER=VTKM_DEVICE_ADAPTER_SERIAL")
  target_compile_options(ExtractTrainingData PRIVATE ${VTKm_COMPILE_OPTIONS})
