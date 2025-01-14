# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Check that the options are properly set for
# the fuzzing project
if (onnxruntime_FUZZ_ENABLED)
	message(STATUS "Building dependency protobuf-mutator and libfuzzer")

	# set the options used to control the protobuf-mutator build
	set(PROTOBUF_LIBRARIES ${PROTOBUF_LIB})
	set(LIB_PROTO_MUTATOR_TESTING OFF)

	# include the protobuf-mutator CMakeLists.txt rather than the projects CMakeLists.txt to avoid target clashes
	# with google test
	add_subdirectory("external/libprotobuf-mutator/src")

	# add the appropriate include directory and compilation flags
	# needed by the protobuf-mutator target and the libfuzzer
	set(PROTOBUF_MUT_INCLUDE_DIRS "external/libprotobuf-mutator")
	onnxruntime_add_include_to_target(protobuf-mutator ${PROTOBUF_LIB})
	onnxruntime_add_include_to_target(protobuf-mutator-libfuzzer ${PROTOBUF_LIB})
	target_include_directories(protobuf-mutator PRIVATE ${INCLUDE_DIRECTORIES} ${PROTOBUF_MUT_INCLUDE_DIRS})
	target_include_directories(protobuf-mutator-libfuzzer PRIVATE ${INCLUDE_DIRECTORIES} ${PROTOBUF_MUT_INCLUDE_DIRS})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # MSVC-specific compiler options
        target_compile_options(protobuf-mutator PRIVATE "/wd4244" "/wd4245" "/wd4267" "/wd4100" "/wd4456")
        target_compile_options(protobuf-mutator-libfuzzer PRIVATE "/wd4146" "/wd4267")
    else()
        # Linux-specific compiler options
        target_compile_options(protobuf-mutator PRIVATE
            -Wno-shorten-64-to-32
            -Wno-conversion
            -Wno-sign-compare
            -Wno-unused-parameter
            -Wno-shadow
            -Wno-unused
            -fexceptions
        )
        target_compile_options(protobuf-mutator-libfuzzer PRIVATE
            -Wno-shorten-64-to-32
            -Wno-conversion
            -Wno-unused
            -fexceptions
        )
    endif()

	# add Fuzzing Engine Build Configuration
	message(STATUS "Building Fuzzing engine")

	# set Fuzz root directory
	set(SEC_FUZZ_ROOT ${TEST_SRC_DIR}/fuzzing)

	# Security fuzzing engine src file reference
	set(SEC_FUZ_SRC "${SEC_FUZZ_ROOT}/src/BetaDistribution.cpp"
					"${SEC_FUZZ_ROOT}/src/OnnxPrediction.cpp"
					"${SEC_FUZZ_ROOT}/src/testlog.cpp"
					"${SEC_FUZZ_ROOT}/src/test.cpp")

	# compile the executables
	onnxruntime_add_executable(onnxruntime_security_fuzz ${SEC_FUZ_SRC})

	# compile with c++17
	target_compile_features(onnxruntime_security_fuzz PUBLIC cxx_std_17)

	# Security fuzzing engine header file reference
	onnxruntime_add_include_to_target(onnxruntime_security_fuzz onnx onnxruntime)

	# Assign all include to one variable
	set(SEC_FUZ_INC "${SEC_FUZZ_ROOT}/include")
	set(INCLUDE_FILES ${SEC_FUZ_INC} "$<TARGET_PROPERTY:protobuf-mutator,INCLUDE_DIRECTORIES>")

	# add all these include directory to the Fuzzing engine
	target_include_directories(onnxruntime_security_fuzz PRIVATE ${INCLUDE_FILES})

	# add link libraries the project
	target_link_libraries(onnxruntime_security_fuzz onnx_proto onnxruntime protobuf-mutator ${PROTOBUF_LIB})

	# add the dependencies
	add_dependencies(onnxruntime_security_fuzz onnx_proto onnxruntime protobuf-mutator ${PROTOBUF_LIB})

	# copy the dlls to the execution directory
	add_custom_command(TARGET onnxruntime_security_fuzz POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:onnxruntime>  $<TARGET_FILE_DIR:onnxruntime_security_fuzz>
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${PROTOBUF_LIB}>  $<TARGET_FILE_DIR:onnxruntime_security_fuzz>)
endif()
