#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/Compose.o \
	${OBJECTDIR}/MultiNet.o \
	${OBJECTDIR}/NetGen.o \
	${OBJECTDIR}/OneRun.o \
	${OBJECTDIR}/SelFail.o \
	${OBJECTDIR}/WordEmbed.o \
	${OBJECTDIR}/caffe/blob.o \
	${OBJECTDIR}/caffe/common.o \
	${OBJECTDIR}/caffe/data_reader.o \
	${OBJECTDIR}/caffe/data_transformer.o \
	${OBJECTDIR}/caffe/internal_thread.o \
	${OBJECTDIR}/caffe/layer.o \
	${OBJECTDIR}/caffe/layer_factory.o \
	${OBJECTDIR}/caffe/layers/absval_layer.o \
	${OBJECTDIR}/caffe/layers/accuracy_layer.o \
	${OBJECTDIR}/caffe/layers/argmax_layer.o \
	${OBJECTDIR}/caffe/layers/base_conv_layer.o \
	${OBJECTDIR}/caffe/layers/base_data_layer.o \
	${OBJECTDIR}/caffe/layers/bnll_layer.o \
	${OBJECTDIR}/caffe/layers/concat_layer.o \
	${OBJECTDIR}/caffe/layers/contrastive_loss_layer.o \
	${OBJECTDIR}/caffe/layers/conv_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_conv_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_pooling_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_relu_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_sigmoid_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_softmax_layer.o \
	${OBJECTDIR}/caffe/layers/cudnn_tanh_layer.o \
	${OBJECTDIR}/caffe/layers/data_layer.o \
	${OBJECTDIR}/caffe/layers/deconv_layer.o \
	${OBJECTDIR}/caffe/layers/dropout_layer.o \
	${OBJECTDIR}/caffe/layers/dummy_data_layer.o \
	${OBJECTDIR}/caffe/layers/eltwise_layer.o \
	${OBJECTDIR}/caffe/layers/embed_layer.o \
	${OBJECTDIR}/caffe/layers/euclidean_loss_layer.o \
	${OBJECTDIR}/caffe/layers/exp_layer.o \
	${OBJECTDIR}/caffe/layers/filter_layer.o \
	${OBJECTDIR}/caffe/layers/flatten_layer.o \
	${OBJECTDIR}/caffe/layers/hdf5_data_layer.o \
	${OBJECTDIR}/caffe/layers/hdf5_output_layer.o \
	${OBJECTDIR}/caffe/layers/hinge_loss_layer.o \
	${OBJECTDIR}/caffe/layers/im2col_layer.o \
	${OBJECTDIR}/caffe/layers/image_data_layer.o \
	${OBJECTDIR}/caffe/layers/infogain_loss_layer.o \
	${OBJECTDIR}/caffe/layers/inner_product_layer.o \
	${OBJECTDIR}/caffe/layers/log_layer.o \
	${OBJECTDIR}/caffe/layers/loss_layer.o \
	${OBJECTDIR}/caffe/layers/lrn_layer.o \
	${OBJECTDIR}/caffe/layers/lstm_layer.o \
	${OBJECTDIR}/caffe/layers/lstm_unit_layer.o \
	${OBJECTDIR}/caffe/layers/memory_data_layer.o \
	${OBJECTDIR}/caffe/layers/multinomial_logistic_loss_layer.o \
	${OBJECTDIR}/caffe/layers/mvn_layer.o \
	${OBJECTDIR}/caffe/layers/neuron_layer.o \
	${OBJECTDIR}/caffe/layers/pooling_layer.o \
	${OBJECTDIR}/caffe/layers/power_layer.o \
	${OBJECTDIR}/caffe/layers/prelu_layer.o \
	${OBJECTDIR}/caffe/layers/recurrent_layer.o \
	${OBJECTDIR}/caffe/layers/reduction_layer.o \
	${OBJECTDIR}/caffe/layers/relu_layer.o \
	${OBJECTDIR}/caffe/layers/reshape_layer.o \
	${OBJECTDIR}/caffe/layers/rnn_layer.o \
	${OBJECTDIR}/caffe/layers/scalar_layer.o \
	${OBJECTDIR}/caffe/layers/sigmoid_cross_entropy_loss_layer.o \
	${OBJECTDIR}/caffe/layers/sigmoid_layer.o \
	${OBJECTDIR}/caffe/layers/silence_layer.o \
	${OBJECTDIR}/caffe/layers/slice_layer.o \
	${OBJECTDIR}/caffe/layers/softmax_layer.o \
	${OBJECTDIR}/caffe/layers/softmax_loss_layer.o \
	${OBJECTDIR}/caffe/layers/split_layer.o \
	${OBJECTDIR}/caffe/layers/spp_layer.o \
	${OBJECTDIR}/caffe/layers/tanh_layer.o \
	${OBJECTDIR}/caffe/layers/threshold_layer.o \
	${OBJECTDIR}/caffe/layers/tile_layer.o \
	${OBJECTDIR}/caffe/layers/window_data_layer.o \
	${OBJECTDIR}/caffe/net.o \
	${OBJECTDIR}/caffe/proto/caffe.pb.o \
	${OBJECTDIR}/caffe/solver.o \
	${OBJECTDIR}/caffe/syncedmem.o \
	${OBJECTDIR}/caffe/test/test_accuracy_layer.o \
	${OBJECTDIR}/caffe/test/test_argmax_layer.o \
	${OBJECTDIR}/caffe/test/test_benchmark.o \
	${OBJECTDIR}/caffe/test/test_blob.o \
	${OBJECTDIR}/caffe/test/test_caffe_main.o \
	${OBJECTDIR}/caffe/test/test_common.o \
	${OBJECTDIR}/caffe/test/test_concat_layer.o \
	${OBJECTDIR}/caffe/test/test_contrastive_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_convolution_layer.o \
	${OBJECTDIR}/caffe/test/test_data_layer.o \
	${OBJECTDIR}/caffe/test/test_data_transformer.o \
	${OBJECTDIR}/caffe/test/test_db.o \
	${OBJECTDIR}/caffe/test/test_deconvolution_layer.o \
	${OBJECTDIR}/caffe/test/test_dummy_data_layer.o \
	${OBJECTDIR}/caffe/test/test_eltwise_layer.o \
	${OBJECTDIR}/caffe/test/test_euclidean_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_filler.o \
	${OBJECTDIR}/caffe/test/test_filter_layer.o \
	${OBJECTDIR}/caffe/test/test_flatten_layer.o \
	${OBJECTDIR}/caffe/test/test_gradient_based_solver.o \
	${OBJECTDIR}/caffe/test/test_hdf5_output_layer.o \
	${OBJECTDIR}/caffe/test/test_hdf5data_layer.o \
	${OBJECTDIR}/caffe/test/test_hinge_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_im2col_layer.o \
	${OBJECTDIR}/caffe/test/test_image_data_layer.o \
	${OBJECTDIR}/caffe/test/test_infogain_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_inner_product_layer.o \
	${OBJECTDIR}/caffe/test/test_internal_thread.o \
	${OBJECTDIR}/caffe/test/test_io.o \
	${OBJECTDIR}/caffe/test/test_layer_factory.o \
	${OBJECTDIR}/caffe/test/test_lrn_layer.o \
	${OBJECTDIR}/caffe/test/test_math_functions.o \
	${OBJECTDIR}/caffe/test/test_maxpool_dropout_layers.o \
	${OBJECTDIR}/caffe/test/test_memory_data_layer.o \
	${OBJECTDIR}/caffe/test/test_multinomial_logistic_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_mvn_layer.o \
	${OBJECTDIR}/caffe/test/test_net.o \
	${OBJECTDIR}/caffe/test/test_neuron_layer.o \
	${OBJECTDIR}/caffe/test/test_platform.o \
	${OBJECTDIR}/caffe/test/test_pooling_layer.o \
	${OBJECTDIR}/caffe/test/test_power_layer.o \
	${OBJECTDIR}/caffe/test/test_protobuf.o \
	${OBJECTDIR}/caffe/test/test_random_number_generator.o \
	${OBJECTDIR}/caffe/test/test_reduction_layer.o \
	${OBJECTDIR}/caffe/test/test_reshape_layer.o \
	${OBJECTDIR}/caffe/test/test_sigmoid_cross_entropy_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_slice_layer.o \
	${OBJECTDIR}/caffe/test/test_softmax_layer.o \
	${OBJECTDIR}/caffe/test/test_softmax_with_loss_layer.o \
	${OBJECTDIR}/caffe/test/test_solver.o \
	${OBJECTDIR}/caffe/test/test_split_layer.o \
	${OBJECTDIR}/caffe/test/test_spp_layer.o \
	${OBJECTDIR}/caffe/test/test_stochastic_pooling.o \
	${OBJECTDIR}/caffe/test/test_syncedmem.o \
	${OBJECTDIR}/caffe/test/test_tanh_layer.o \
	${OBJECTDIR}/caffe/test/test_threshold_layer.o \
	${OBJECTDIR}/caffe/test/test_upgrade_proto.o \
	${OBJECTDIR}/caffe/test/test_util_blas.o \
	${OBJECTDIR}/caffe/util/benchmark.o \
	${OBJECTDIR}/caffe/util/blocking_queue.o \
	${OBJECTDIR}/caffe/util/cudnn.o \
	${OBJECTDIR}/caffe/util/db.o \
	${OBJECTDIR}/caffe/util/db_leveldb.o \
	${OBJECTDIR}/caffe/util/db_lmdb.o \
	${OBJECTDIR}/caffe/util/hdf5.o \
	${OBJECTDIR}/caffe/util/im2col.o \
	${OBJECTDIR}/caffe/util/insert_splits.o \
	${OBJECTDIR}/caffe/util/io.o \
	${OBJECTDIR}/caffe/util/math_functions.o \
	${OBJECTDIR}/caffe/util/signal_handler.o \
	${OBJECTDIR}/caffe/util/upgrade_proto.o \
	${OBJECTDIR}/classification.o \
	${OBJECTDIR}/gtest/gtest-all.o \
	${OBJECTDIR}/ipc.pb.o \
	${OBJECTDIR}/tools/caffe.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-pthread -fPIC
CXXFLAGS=-pthread -fPIC

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-lglog -lgflags -lprotobuf -llmdb -lopencv_core -lopencv_imgproc -lopencv_highgui -lhdf5 -lhdf5_hl -lboost_system -lboost_thread -lleveldb -lsnappy -lm -lstdc++ -lcblas -latlas -lpthread -lhdf5_cpp

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/caffer

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/caffer: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/caffer ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/Compose.o: Compose.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Compose.o Compose.cpp

${OBJECTDIR}/MultiNet.o: MultiNet.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MultiNet.o MultiNet.cpp

${OBJECTDIR}/NetGen.o: NetGen.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/NetGen.o NetGen.cpp

${OBJECTDIR}/OneRun.o: OneRun.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/OneRun.o OneRun.cpp

${OBJECTDIR}/SelFail.o: SelFail.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/SelFail.o SelFail.cpp

${OBJECTDIR}/WordEmbed.o: WordEmbed.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/WordEmbed.o WordEmbed.cpp

${OBJECTDIR}/caffe/blob.o: caffe/blob.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/blob.o caffe/blob.cpp

${OBJECTDIR}/caffe/common.o: caffe/common.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/common.o caffe/common.cpp

${OBJECTDIR}/caffe/data_reader.o: caffe/data_reader.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/data_reader.o caffe/data_reader.cpp

${OBJECTDIR}/caffe/data_transformer.o: caffe/data_transformer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/data_transformer.o caffe/data_transformer.cpp

${OBJECTDIR}/caffe/internal_thread.o: caffe/internal_thread.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/internal_thread.o caffe/internal_thread.cpp

${OBJECTDIR}/caffe/layer.o: caffe/layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layer.o caffe/layer.cpp

${OBJECTDIR}/caffe/layer_factory.o: caffe/layer_factory.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layer_factory.o caffe/layer_factory.cpp

${OBJECTDIR}/caffe/layers/absval_layer.o: caffe/layers/absval_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/absval_layer.o caffe/layers/absval_layer.cpp

${OBJECTDIR}/caffe/layers/accuracy_layer.o: caffe/layers/accuracy_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/accuracy_layer.o caffe/layers/accuracy_layer.cpp

${OBJECTDIR}/caffe/layers/argmax_layer.o: caffe/layers/argmax_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/argmax_layer.o caffe/layers/argmax_layer.cpp

${OBJECTDIR}/caffe/layers/base_conv_layer.o: caffe/layers/base_conv_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/base_conv_layer.o caffe/layers/base_conv_layer.cpp

${OBJECTDIR}/caffe/layers/base_data_layer.o: caffe/layers/base_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/base_data_layer.o caffe/layers/base_data_layer.cpp

${OBJECTDIR}/caffe/layers/bnll_layer.o: caffe/layers/bnll_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/bnll_layer.o caffe/layers/bnll_layer.cpp

${OBJECTDIR}/caffe/layers/concat_layer.o: caffe/layers/concat_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/concat_layer.o caffe/layers/concat_layer.cpp

${OBJECTDIR}/caffe/layers/contrastive_loss_layer.o: caffe/layers/contrastive_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/contrastive_loss_layer.o caffe/layers/contrastive_loss_layer.cpp

${OBJECTDIR}/caffe/layers/conv_layer.o: caffe/layers/conv_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/conv_layer.o caffe/layers/conv_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_conv_layer.o: caffe/layers/cudnn_conv_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_conv_layer.o caffe/layers/cudnn_conv_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_pooling_layer.o: caffe/layers/cudnn_pooling_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_pooling_layer.o caffe/layers/cudnn_pooling_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_relu_layer.o: caffe/layers/cudnn_relu_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_relu_layer.o caffe/layers/cudnn_relu_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_sigmoid_layer.o: caffe/layers/cudnn_sigmoid_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_sigmoid_layer.o caffe/layers/cudnn_sigmoid_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_softmax_layer.o: caffe/layers/cudnn_softmax_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_softmax_layer.o caffe/layers/cudnn_softmax_layer.cpp

${OBJECTDIR}/caffe/layers/cudnn_tanh_layer.o: caffe/layers/cudnn_tanh_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/cudnn_tanh_layer.o caffe/layers/cudnn_tanh_layer.cpp

${OBJECTDIR}/caffe/layers/data_layer.o: caffe/layers/data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/data_layer.o caffe/layers/data_layer.cpp

${OBJECTDIR}/caffe/layers/deconv_layer.o: caffe/layers/deconv_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/deconv_layer.o caffe/layers/deconv_layer.cpp

${OBJECTDIR}/caffe/layers/dropout_layer.o: caffe/layers/dropout_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/dropout_layer.o caffe/layers/dropout_layer.cpp

${OBJECTDIR}/caffe/layers/dummy_data_layer.o: caffe/layers/dummy_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/dummy_data_layer.o caffe/layers/dummy_data_layer.cpp

${OBJECTDIR}/caffe/layers/eltwise_layer.o: caffe/layers/eltwise_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/eltwise_layer.o caffe/layers/eltwise_layer.cpp

${OBJECTDIR}/caffe/layers/embed_layer.o: caffe/layers/embed_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/embed_layer.o caffe/layers/embed_layer.cpp

${OBJECTDIR}/caffe/layers/euclidean_loss_layer.o: caffe/layers/euclidean_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/euclidean_loss_layer.o caffe/layers/euclidean_loss_layer.cpp

${OBJECTDIR}/caffe/layers/exp_layer.o: caffe/layers/exp_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/exp_layer.o caffe/layers/exp_layer.cpp

${OBJECTDIR}/caffe/layers/filter_layer.o: caffe/layers/filter_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/filter_layer.o caffe/layers/filter_layer.cpp

${OBJECTDIR}/caffe/layers/flatten_layer.o: caffe/layers/flatten_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/flatten_layer.o caffe/layers/flatten_layer.cpp

${OBJECTDIR}/caffe/layers/hdf5_data_layer.o: caffe/layers/hdf5_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/hdf5_data_layer.o caffe/layers/hdf5_data_layer.cpp

${OBJECTDIR}/caffe/layers/hdf5_output_layer.o: caffe/layers/hdf5_output_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/hdf5_output_layer.o caffe/layers/hdf5_output_layer.cpp

${OBJECTDIR}/caffe/layers/hinge_loss_layer.o: caffe/layers/hinge_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/hinge_loss_layer.o caffe/layers/hinge_loss_layer.cpp

${OBJECTDIR}/caffe/layers/im2col_layer.o: caffe/layers/im2col_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/im2col_layer.o caffe/layers/im2col_layer.cpp

${OBJECTDIR}/caffe/layers/image_data_layer.o: caffe/layers/image_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/image_data_layer.o caffe/layers/image_data_layer.cpp

${OBJECTDIR}/caffe/layers/infogain_loss_layer.o: caffe/layers/infogain_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/infogain_loss_layer.o caffe/layers/infogain_loss_layer.cpp

${OBJECTDIR}/caffe/layers/inner_product_layer.o: caffe/layers/inner_product_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/inner_product_layer.o caffe/layers/inner_product_layer.cpp

${OBJECTDIR}/caffe/layers/log_layer.o: caffe/layers/log_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/log_layer.o caffe/layers/log_layer.cpp

${OBJECTDIR}/caffe/layers/loss_layer.o: caffe/layers/loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/loss_layer.o caffe/layers/loss_layer.cpp

${OBJECTDIR}/caffe/layers/lrn_layer.o: caffe/layers/lrn_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/lrn_layer.o caffe/layers/lrn_layer.cpp

${OBJECTDIR}/caffe/layers/lstm_layer.o: caffe/layers/lstm_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/lstm_layer.o caffe/layers/lstm_layer.cpp

${OBJECTDIR}/caffe/layers/lstm_unit_layer.o: caffe/layers/lstm_unit_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/lstm_unit_layer.o caffe/layers/lstm_unit_layer.cpp

${OBJECTDIR}/caffe/layers/memory_data_layer.o: caffe/layers/memory_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/memory_data_layer.o caffe/layers/memory_data_layer.cpp

${OBJECTDIR}/caffe/layers/multinomial_logistic_loss_layer.o: caffe/layers/multinomial_logistic_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/multinomial_logistic_loss_layer.o caffe/layers/multinomial_logistic_loss_layer.cpp

${OBJECTDIR}/caffe/layers/mvn_layer.o: caffe/layers/mvn_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/mvn_layer.o caffe/layers/mvn_layer.cpp

${OBJECTDIR}/caffe/layers/neuron_layer.o: caffe/layers/neuron_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/neuron_layer.o caffe/layers/neuron_layer.cpp

${OBJECTDIR}/caffe/layers/pooling_layer.o: caffe/layers/pooling_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/pooling_layer.o caffe/layers/pooling_layer.cpp

${OBJECTDIR}/caffe/layers/power_layer.o: caffe/layers/power_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/power_layer.o caffe/layers/power_layer.cpp

${OBJECTDIR}/caffe/layers/prelu_layer.o: caffe/layers/prelu_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/prelu_layer.o caffe/layers/prelu_layer.cpp

${OBJECTDIR}/caffe/layers/recurrent_layer.o: caffe/layers/recurrent_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/recurrent_layer.o caffe/layers/recurrent_layer.cpp

${OBJECTDIR}/caffe/layers/reduction_layer.o: caffe/layers/reduction_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/reduction_layer.o caffe/layers/reduction_layer.cpp

${OBJECTDIR}/caffe/layers/relu_layer.o: caffe/layers/relu_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/relu_layer.o caffe/layers/relu_layer.cpp

${OBJECTDIR}/caffe/layers/reshape_layer.o: caffe/layers/reshape_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/reshape_layer.o caffe/layers/reshape_layer.cpp

${OBJECTDIR}/caffe/layers/rnn_layer.o: caffe/layers/rnn_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/rnn_layer.o caffe/layers/rnn_layer.cpp

${OBJECTDIR}/caffe/layers/scalar_layer.o: caffe/layers/scalar_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/scalar_layer.o caffe/layers/scalar_layer.cpp

${OBJECTDIR}/caffe/layers/sigmoid_cross_entropy_loss_layer.o: caffe/layers/sigmoid_cross_entropy_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/sigmoid_cross_entropy_loss_layer.o caffe/layers/sigmoid_cross_entropy_loss_layer.cpp

${OBJECTDIR}/caffe/layers/sigmoid_layer.o: caffe/layers/sigmoid_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/sigmoid_layer.o caffe/layers/sigmoid_layer.cpp

${OBJECTDIR}/caffe/layers/silence_layer.o: caffe/layers/silence_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/silence_layer.o caffe/layers/silence_layer.cpp

${OBJECTDIR}/caffe/layers/slice_layer.o: caffe/layers/slice_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/slice_layer.o caffe/layers/slice_layer.cpp

${OBJECTDIR}/caffe/layers/softmax_layer.o: caffe/layers/softmax_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/softmax_layer.o caffe/layers/softmax_layer.cpp

${OBJECTDIR}/caffe/layers/softmax_loss_layer.o: caffe/layers/softmax_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/softmax_loss_layer.o caffe/layers/softmax_loss_layer.cpp

${OBJECTDIR}/caffe/layers/split_layer.o: caffe/layers/split_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/split_layer.o caffe/layers/split_layer.cpp

${OBJECTDIR}/caffe/layers/spp_layer.o: caffe/layers/spp_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/spp_layer.o caffe/layers/spp_layer.cpp

${OBJECTDIR}/caffe/layers/tanh_layer.o: caffe/layers/tanh_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/tanh_layer.o caffe/layers/tanh_layer.cpp

${OBJECTDIR}/caffe/layers/threshold_layer.o: caffe/layers/threshold_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/threshold_layer.o caffe/layers/threshold_layer.cpp

${OBJECTDIR}/caffe/layers/tile_layer.o: caffe/layers/tile_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/tile_layer.o caffe/layers/tile_layer.cpp

${OBJECTDIR}/caffe/layers/window_data_layer.o: caffe/layers/window_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/layers
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/layers/window_data_layer.o caffe/layers/window_data_layer.cpp

${OBJECTDIR}/caffe/net.o: caffe/net.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/net.o caffe/net.cpp

${OBJECTDIR}/caffe/proto/caffe.pb.o: caffe/proto/caffe.pb.cc 
	${MKDIR} -p ${OBJECTDIR}/caffe/proto
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/proto/caffe.pb.o caffe/proto/caffe.pb.cc

${OBJECTDIR}/caffe/solver.o: caffe/solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/solver.o caffe/solver.cpp

${OBJECTDIR}/caffe/syncedmem.o: caffe/syncedmem.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/syncedmem.o caffe/syncedmem.cpp

${OBJECTDIR}/caffe/test/test_accuracy_layer.o: caffe/test/test_accuracy_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_accuracy_layer.o caffe/test/test_accuracy_layer.cpp

${OBJECTDIR}/caffe/test/test_argmax_layer.o: caffe/test/test_argmax_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_argmax_layer.o caffe/test/test_argmax_layer.cpp

${OBJECTDIR}/caffe/test/test_benchmark.o: caffe/test/test_benchmark.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_benchmark.o caffe/test/test_benchmark.cpp

${OBJECTDIR}/caffe/test/test_blob.o: caffe/test/test_blob.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_blob.o caffe/test/test_blob.cpp

${OBJECTDIR}/caffe/test/test_caffe_main.o: caffe/test/test_caffe_main.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_caffe_main.o caffe/test/test_caffe_main.cpp

${OBJECTDIR}/caffe/test/test_common.o: caffe/test/test_common.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_common.o caffe/test/test_common.cpp

${OBJECTDIR}/caffe/test/test_concat_layer.o: caffe/test/test_concat_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_concat_layer.o caffe/test/test_concat_layer.cpp

${OBJECTDIR}/caffe/test/test_contrastive_loss_layer.o: caffe/test/test_contrastive_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_contrastive_loss_layer.o caffe/test/test_contrastive_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_convolution_layer.o: caffe/test/test_convolution_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_convolution_layer.o caffe/test/test_convolution_layer.cpp

${OBJECTDIR}/caffe/test/test_data_layer.o: caffe/test/test_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_data_layer.o caffe/test/test_data_layer.cpp

${OBJECTDIR}/caffe/test/test_data_transformer.o: caffe/test/test_data_transformer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_data_transformer.o caffe/test/test_data_transformer.cpp

${OBJECTDIR}/caffe/test/test_db.o: caffe/test/test_db.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_db.o caffe/test/test_db.cpp

${OBJECTDIR}/caffe/test/test_deconvolution_layer.o: caffe/test/test_deconvolution_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_deconvolution_layer.o caffe/test/test_deconvolution_layer.cpp

${OBJECTDIR}/caffe/test/test_dummy_data_layer.o: caffe/test/test_dummy_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_dummy_data_layer.o caffe/test/test_dummy_data_layer.cpp

${OBJECTDIR}/caffe/test/test_eltwise_layer.o: caffe/test/test_eltwise_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_eltwise_layer.o caffe/test/test_eltwise_layer.cpp

${OBJECTDIR}/caffe/test/test_euclidean_loss_layer.o: caffe/test/test_euclidean_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_euclidean_loss_layer.o caffe/test/test_euclidean_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_filler.o: caffe/test/test_filler.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_filler.o caffe/test/test_filler.cpp

${OBJECTDIR}/caffe/test/test_filter_layer.o: caffe/test/test_filter_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_filter_layer.o caffe/test/test_filter_layer.cpp

${OBJECTDIR}/caffe/test/test_flatten_layer.o: caffe/test/test_flatten_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_flatten_layer.o caffe/test/test_flatten_layer.cpp

${OBJECTDIR}/caffe/test/test_gradient_based_solver.o: caffe/test/test_gradient_based_solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_gradient_based_solver.o caffe/test/test_gradient_based_solver.cpp

${OBJECTDIR}/caffe/test/test_hdf5_output_layer.o: caffe/test/test_hdf5_output_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_hdf5_output_layer.o caffe/test/test_hdf5_output_layer.cpp

${OBJECTDIR}/caffe/test/test_hdf5data_layer.o: caffe/test/test_hdf5data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_hdf5data_layer.o caffe/test/test_hdf5data_layer.cpp

${OBJECTDIR}/caffe/test/test_hinge_loss_layer.o: caffe/test/test_hinge_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_hinge_loss_layer.o caffe/test/test_hinge_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_im2col_layer.o: caffe/test/test_im2col_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_im2col_layer.o caffe/test/test_im2col_layer.cpp

${OBJECTDIR}/caffe/test/test_image_data_layer.o: caffe/test/test_image_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_image_data_layer.o caffe/test/test_image_data_layer.cpp

${OBJECTDIR}/caffe/test/test_infogain_loss_layer.o: caffe/test/test_infogain_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_infogain_loss_layer.o caffe/test/test_infogain_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_inner_product_layer.o: caffe/test/test_inner_product_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_inner_product_layer.o caffe/test/test_inner_product_layer.cpp

${OBJECTDIR}/caffe/test/test_internal_thread.o: caffe/test/test_internal_thread.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_internal_thread.o caffe/test/test_internal_thread.cpp

${OBJECTDIR}/caffe/test/test_io.o: caffe/test/test_io.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_io.o caffe/test/test_io.cpp

${OBJECTDIR}/caffe/test/test_layer_factory.o: caffe/test/test_layer_factory.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_layer_factory.o caffe/test/test_layer_factory.cpp

${OBJECTDIR}/caffe/test/test_lrn_layer.o: caffe/test/test_lrn_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_lrn_layer.o caffe/test/test_lrn_layer.cpp

${OBJECTDIR}/caffe/test/test_math_functions.o: caffe/test/test_math_functions.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_math_functions.o caffe/test/test_math_functions.cpp

${OBJECTDIR}/caffe/test/test_maxpool_dropout_layers.o: caffe/test/test_maxpool_dropout_layers.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_maxpool_dropout_layers.o caffe/test/test_maxpool_dropout_layers.cpp

${OBJECTDIR}/caffe/test/test_memory_data_layer.o: caffe/test/test_memory_data_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_memory_data_layer.o caffe/test/test_memory_data_layer.cpp

${OBJECTDIR}/caffe/test/test_multinomial_logistic_loss_layer.o: caffe/test/test_multinomial_logistic_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_multinomial_logistic_loss_layer.o caffe/test/test_multinomial_logistic_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_mvn_layer.o: caffe/test/test_mvn_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_mvn_layer.o caffe/test/test_mvn_layer.cpp

${OBJECTDIR}/caffe/test/test_net.o: caffe/test/test_net.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_net.o caffe/test/test_net.cpp

${OBJECTDIR}/caffe/test/test_neuron_layer.o: caffe/test/test_neuron_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_neuron_layer.o caffe/test/test_neuron_layer.cpp

${OBJECTDIR}/caffe/test/test_platform.o: caffe/test/test_platform.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_platform.o caffe/test/test_platform.cpp

${OBJECTDIR}/caffe/test/test_pooling_layer.o: caffe/test/test_pooling_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_pooling_layer.o caffe/test/test_pooling_layer.cpp

${OBJECTDIR}/caffe/test/test_power_layer.o: caffe/test/test_power_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_power_layer.o caffe/test/test_power_layer.cpp

${OBJECTDIR}/caffe/test/test_protobuf.o: caffe/test/test_protobuf.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_protobuf.o caffe/test/test_protobuf.cpp

${OBJECTDIR}/caffe/test/test_random_number_generator.o: caffe/test/test_random_number_generator.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_random_number_generator.o caffe/test/test_random_number_generator.cpp

${OBJECTDIR}/caffe/test/test_reduction_layer.o: caffe/test/test_reduction_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_reduction_layer.o caffe/test/test_reduction_layer.cpp

${OBJECTDIR}/caffe/test/test_reshape_layer.o: caffe/test/test_reshape_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_reshape_layer.o caffe/test/test_reshape_layer.cpp

${OBJECTDIR}/caffe/test/test_sigmoid_cross_entropy_loss_layer.o: caffe/test/test_sigmoid_cross_entropy_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_sigmoid_cross_entropy_loss_layer.o caffe/test/test_sigmoid_cross_entropy_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_slice_layer.o: caffe/test/test_slice_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_slice_layer.o caffe/test/test_slice_layer.cpp

${OBJECTDIR}/caffe/test/test_softmax_layer.o: caffe/test/test_softmax_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_softmax_layer.o caffe/test/test_softmax_layer.cpp

${OBJECTDIR}/caffe/test/test_softmax_with_loss_layer.o: caffe/test/test_softmax_with_loss_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_softmax_with_loss_layer.o caffe/test/test_softmax_with_loss_layer.cpp

${OBJECTDIR}/caffe/test/test_solver.o: caffe/test/test_solver.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_solver.o caffe/test/test_solver.cpp

${OBJECTDIR}/caffe/test/test_split_layer.o: caffe/test/test_split_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_split_layer.o caffe/test/test_split_layer.cpp

${OBJECTDIR}/caffe/test/test_spp_layer.o: caffe/test/test_spp_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_spp_layer.o caffe/test/test_spp_layer.cpp

${OBJECTDIR}/caffe/test/test_stochastic_pooling.o: caffe/test/test_stochastic_pooling.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_stochastic_pooling.o caffe/test/test_stochastic_pooling.cpp

${OBJECTDIR}/caffe/test/test_syncedmem.o: caffe/test/test_syncedmem.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_syncedmem.o caffe/test/test_syncedmem.cpp

${OBJECTDIR}/caffe/test/test_tanh_layer.o: caffe/test/test_tanh_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_tanh_layer.o caffe/test/test_tanh_layer.cpp

${OBJECTDIR}/caffe/test/test_threshold_layer.o: caffe/test/test_threshold_layer.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_threshold_layer.o caffe/test/test_threshold_layer.cpp

${OBJECTDIR}/caffe/test/test_upgrade_proto.o: caffe/test/test_upgrade_proto.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_upgrade_proto.o caffe/test/test_upgrade_proto.cpp

${OBJECTDIR}/caffe/test/test_util_blas.o: caffe/test/test_util_blas.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/test
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/test/test_util_blas.o caffe/test/test_util_blas.cpp

${OBJECTDIR}/caffe/util/benchmark.o: caffe/util/benchmark.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/benchmark.o caffe/util/benchmark.cpp

${OBJECTDIR}/caffe/util/blocking_queue.o: caffe/util/blocking_queue.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/blocking_queue.o caffe/util/blocking_queue.cpp

${OBJECTDIR}/caffe/util/cudnn.o: caffe/util/cudnn.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/cudnn.o caffe/util/cudnn.cpp

${OBJECTDIR}/caffe/util/db.o: caffe/util/db.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/db.o caffe/util/db.cpp

${OBJECTDIR}/caffe/util/db_leveldb.o: caffe/util/db_leveldb.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/db_leveldb.o caffe/util/db_leveldb.cpp

${OBJECTDIR}/caffe/util/db_lmdb.o: caffe/util/db_lmdb.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/db_lmdb.o caffe/util/db_lmdb.cpp

${OBJECTDIR}/caffe/util/hdf5.o: caffe/util/hdf5.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/hdf5.o caffe/util/hdf5.cpp

${OBJECTDIR}/caffe/util/im2col.o: caffe/util/im2col.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/im2col.o caffe/util/im2col.cpp

${OBJECTDIR}/caffe/util/insert_splits.o: caffe/util/insert_splits.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/insert_splits.o caffe/util/insert_splits.cpp

${OBJECTDIR}/caffe/util/io.o: caffe/util/io.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/io.o caffe/util/io.cpp

${OBJECTDIR}/caffe/util/math_functions.o: caffe/util/math_functions.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/math_functions.o caffe/util/math_functions.cpp

${OBJECTDIR}/caffe/util/signal_handler.o: caffe/util/signal_handler.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/signal_handler.o caffe/util/signal_handler.cpp

${OBJECTDIR}/caffe/util/upgrade_proto.o: caffe/util/upgrade_proto.cpp 
	${MKDIR} -p ${OBJECTDIR}/caffe/util
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/caffe/util/upgrade_proto.o caffe/util/upgrade_proto.cpp

${OBJECTDIR}/classification.o: classification.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/classification.o classification.cpp

${OBJECTDIR}/gtest/gtest-all.o: gtest/gtest-all.cpp 
	${MKDIR} -p ${OBJECTDIR}/gtest
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/gtest/gtest-all.o gtest/gtest-all.cpp

${OBJECTDIR}/ipc.pb.o: ipc.pb.cc 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/ipc.pb.o ipc.pb.cc

${OBJECTDIR}/tools/caffe.o: tools/caffe.cpp 
	${MKDIR} -p ${OBJECTDIR}/tools
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -DCAFFE_COMPOSE_MAIN -DCPU_ONLY -I../../caffe-recurrent/build/src -I. -Iinclude -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/tools/caffe.o tools/caffe.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/caffer

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
