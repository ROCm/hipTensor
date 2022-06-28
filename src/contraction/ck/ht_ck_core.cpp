#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_contraction_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"
#include "ht_types.hpp"
#include "ht_ck_core.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
// Fast changing dimension in A/B/C are K/N/N dimensions
using ContractionInstanceKNN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are K/K/N dimensions
using ContractionInstanceKKN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are M/N/N dimensions
using ContractionInstanceMNN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   1,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are M/K/N dimensions
using ContractionInstanceMKN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   1,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,               S<1, 16, 1, 16>,              4>;
// clang-format on


char hiptensorDeriveFastChangeDimension(const int index, const int ndim, int parse_char_index)
{
    if (parse_char_index == 0)
    {
        if (index > ndim)
            return 'K';
        else
            return 'M';
    }
    else if (parse_char_index == 1)
    {
        if (index > ndim)
            return 'N';

        else
            return 'K';
    }
	else if (parse_char_index == 2)
    {
        if (index > ndim)
            return 'N';

        else
            return 'M';
    }
	else{
		std::cout << "Invalid index/ out of bound array" << std::endl;
	}
	return '\0';
}


hiptensorStatus_t hiptensorDeriveLayoutFromInputs(hiptensorContractionDescriptor_t* desc, const int ndim)
{
	if (!desc)
		return HIPTENSOR_STATUS_NOT_INITIALIZED;

	char fast_changing[4]={0};
	int char_index = 0;

    for(auto it = desc->ht_contract_desc.begin(); it < desc->ht_contract_desc.end(); ++it)
    {
    	auto it1 = std::find(it->strides.begin(), it->strides.end(), 1);
		
		if (it1 != it->strides.end())
		{
			fast_changing[char_index] = hiptensorDeriveFastChangeDimension((it1 - it->strides.begin()), ndim, char_index);
			char_index++;
    	}
	}


	if (!strcmp(fast_changing, "KNN"))
		desc->ht_contract_layout = HIPTENSOR_CONTRACTION_KNN;

	else if(!strcmp(fast_changing, "KKN"))
		desc->ht_contract_layout = HIPTENSOR_CONTRACTION_KKN;

	else if(!strcmp(fast_changing, "MNN"))
		desc->ht_contract_layout = HIPTENSOR_CONTRACTION_MNN;
		
	else
		desc->ht_contract_layout = HIPTENSOR_CONTRACTION_MKN;

	return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCKContraction(const hiptensorHandle_t* handle,
                                     const hiptensorContractionPlan_t* plan, 
									 hiptensorContractionMetrics_t *ht_contract_metrics,
                                     const void* alpha, const void* A, const void* B,
                                     const void* beta,  const void* C,       void* D,
                                     void *workspace, uint64_t workspaceSize, hipStream_t stream)
{
	if (!handle || !ht_contract_metrics || !A || !B || !D)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    
    // device operation
    //std::unique_ptrck::tensor_operation::device::DeviceContraction> op_ptr;
    
/*
    if (plan->ht_plan_desc.ht_contract_layout == HIPTENSOR_CONTRACTION_KNN)
    {
        op_ptr =  std::make_unique<ContractionInstanceKNN>(ContractionInstanceKNN(){});
    }
	else if (plan->ht_plan_desc.ht_contract_layout == HIPTENSOR_CONTRACTION_KKN)
    {
        op_ptr =  std::make_unique<ContractionInstanceKNN>(ContractionInstanceKKN(){});
    }
	else if (plan->ht_plan_desc.ht_contract_layout == HIPTENSOR_CONTRACTION_MNN)
    {
        op_ptr =  std::make_unique<ContractionInstanceKNN>(ContractionInstanceMNN(){});
    }
	else
    {
        op_ptr =  std::make_unique<ContractionInstanceKNN>(ContractionInstanceMKN(){});
    }
*/	
    
    auto op_ptr = ContractionInstanceKNN{};
    auto invoker  = op_ptr.MakeInvoker();
    auto argument = op_ptr.MakeArgument((ADataType *) A,
                                    (BDataType *) B,
                                    (CDataType *) D,
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_desc[0].lens.end()),
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_desc[0].strides.end()),
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_desc[1].lens.end()),
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_desc[1].strides.end()),
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_desc[2].lens.end()),
                                    std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_desc[2].strides.end()),
                                    a_element_op,
                                    b_element_op,
                                    c_element_op);
        
    if(!op_ptr.IsSupportedArgument(argument))
    {
        std::cout << op_ptr.GetTypeString() << " does not support this problem" << std::endl;
         return HIPTENSOR_STATUS_SUCCESS;
    }

    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));

    ht_contract_metrics->avg_time = invoker.Run(argument, StreamConfig{nullptr, true});

    ck::index_t M = std::accumulate(plan->ht_plan_desc.ht_contract_desc[2].lens.begin(),
                                    plan->ht_plan_desc.ht_contract_desc[2].lens.begin() + NumDimM,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t N = std::accumulate(plan->ht_plan_desc.ht_contract_desc[2].lens.begin() + NumDimM,
                                    plan->ht_plan_desc.ht_contract_desc[2].lens.begin() + NumDimM + NumDimN,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t K = std::accumulate(plan->ht_plan_desc.ht_contract_desc[0].lens.begin() + NumDimM,
                                    plan->ht_plan_desc.ht_contract_desc[0].lens.begin() + NumDimM + NumDimK,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    ht_contract_metrics->tflops = static_cast<float>(flop) / 1.E9 / ht_contract_metrics->avg_time;
    ht_contract_metrics->transfer_speed = num_btype / 1.E6 / ht_contract_metrics->avg_time;

    return HIPTENSOR_STATUS_SUCCESS;

}
