cmake_minimum_required(VERSION 3.10)
project(test_icp_run_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Support both Debug and Release in multi-config generators (VS/Xcode).
if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Supported build configurations" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    # Keep Release as default only for single-config generators.
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Optional: point these to your local dependency config folders before configure.
set(OpenCV_DIR "D:/personal_code_rep/thirdparty/opencv480/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

set(Open3D_DIR "D:/personal_code_rep/thirdparty/open3d-devel-windows-amd64-0.17.0/CMake") 
find_package(Open3D REQUIRED)


set(PCL_DIR "D:/personal_code_rep/thirdparty/PCL 1.15.1/cmake")
find_package(PCL REQUIRED COMPONENTS
    common
    io
    filters
    search
    kdtree
    features
    registration
    sample_consensus
)

set(Eigen3_DIR "D:/personal_code_rep/thirdparty/eigen-3.4.0/share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable(test_icp_run_standalone
    src/main.cpp
    src/HVCloudPairICP.cpp
    src/ICPCore.cpp
    src/HVUtils.cpp
    src/HVI18n.cpp
)

target_include_directories(test_icp_run_standalone PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/public
    ${OpenCV_INCLUDE_DIRS}
    ${Open3D_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(test_icp_run_standalone PRIVATE
    ${OpenCV_LIBS}
    Open3D::Open3D
    ${PCL_LIBRARIES}
    Eigen3::Eigen
)

if(MSVC)
    target_compile_options(test_icp_run_standalone PRIVATE
        /bigobj
        /utf-8
        $<$<CONFIG:Debug>:/Zi /Od /RTC1>
    )
    set_target_properties(test_icp_run_standalone PROPERTIES
        LINK_FLAGS_DEBUG "/DEBUG"
    )
endif()

if(WIN32)
    set(OpenCV_DLL_DIR "" CACHE PATH "Directory containing OpenCV runtime DLLs")
    if(NOT OpenCV_DLL_DIR)
        get_filename_component(_OPENCV_DIR_ABS "${OpenCV_DIR}" ABSOLUTE)
        set(_OPENCV_DLL_CANDIDATES
            "${_OPENCV_DIR_ABS}/../bin"
            "${_OPENCV_DIR_ABS}/bin"
            "${_OPENCV_DIR_ABS}/../../bin"
            "${_OPENCV_DIR_ABS}/x64/vc16/bin"
            "${_OPENCV_DIR_ABS}/../x64/vc16/bin"
        )
        foreach(_candidate IN LISTS _OPENCV_DLL_CANDIDATES)
            if(EXISTS "${_candidate}")
                file(GLOB _candidate_dlls "${_candidate}/opencv*.dll")
                if(_candidate_dlls)
                    set(OpenCV_DLL_DIR "${_candidate}" CACHE PATH "Directory containing OpenCV runtime DLLs" FORCE)
                    break()
                endif()
            endif()
        endforeach()
    endif()

    if(OpenCV_DLL_DIR)
        file(GLOB OpenCV_DLLS "${OpenCV_DLL_DIR}/opencv*.dll")
        if(OpenCV_DLLS)
            add_custom_command(TARGET test_icp_run_standalone POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OpenCV_DLLS} $<TARGET_FILE_DIR:test_icp_run_standalone>
                COMMAND_EXPAND_LISTS
            )
        endif()
    else()
        message(WARNING "OpenCV runtime DLL directory not found. Set OpenCV_DLL_DIR manually.")
    endif()

    get_filename_component(Open3D_ROOT_DIR "${Open3D_DIR}" DIRECTORY)
    set(Open3D_DLL_DIR "${Open3D_ROOT_DIR}/bin")
    file(GLOB OPEN3D_DLLS "${Open3D_DLL_DIR}/*.dll")
    if(OPEN3D_DLLS)
        add_custom_command(TARGET test_icp_run_standalone POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPEN3D_DLLS} $<TARGET_FILE_DIR:test_icp_run_standalone>
        )
    endif()
endif()
