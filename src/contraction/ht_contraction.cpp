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
#include "ht_tensor.hpp"

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
using DeviceOpInstance = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,              4>;
// clang-format on


hiptensorStatus_t hiptensorInitContractionDescriptor(const hiptensorHandle_t* handle,
                                                    hiptensorContractionDescriptor_t* desc,
                                                    const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
                                                    const hiptensorTensorDescriptor_t* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
                                                    const hiptensorTensorDescriptor_t* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
                                                    const hiptensorTensorDescriptor_t* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
                                                    hiptensorComputeType_t typeCompute)

{
    std::cout << "Entered the " << __func__ << std::endl;
    
    if (!handle || !desc || !descA || !descB || !descC )
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    const hiptensorTensorDescriptor_t *ht_input_descs[] = { descA, descB, descC };
    desc->hiptensorContractionAttrUpdate(ht_input_descs, 3);
    
    std::cout << "Exited the " << __func__ << std::endl;

    return HIPTENSOR_STATUS_SUCCESS;
}


hiptensorStatus_t hiptensorInitContractionFind(const hiptensorHandle_t* handle,
                                             hiptensorContractionFind_t* find,
                                             const hiptensorAlgo_t algo)
{
    std::cout << "Entered the " << __func__ << std::endl;
    std::cout << "Exited the " << __func__ << std::endl;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContractionGetWorkspace(const hiptensorHandle_t* handle,
                                                 const hiptensorContractionDescriptor_t* desc,
                                                 const hiptensorContractionFind_t* find,
                                                 const hiptensorWorksizePreference_t pref,
                                                 uint64_t *workspaceSize) 
{
    std::cout << "Entered the " << __func__ << std::endl;
    std::cout << "Exited the " << __func__ << std::endl;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitContractionPlan(const hiptensorHandle_t* handle,
                                            hiptensorContractionPlan_t* plan, const hiptensorContractionDescriptor_t* desc,
                                            const hiptensorContractionFind_t* find,
                                            const uint64_t workspaceSize) 
{
    std::cout << "Entered the " << __func__ << std::endl;

    if (!handle || !plan || !desc)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    plan->ht_plan_desc = *desc; 
    std::cout << "Exited the " << __func__ << std::endl;
    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t* handle,
                                     const hiptensorContractionPlan_t* plan,
                                     const void* alpha, const void* A, const void* B,
                                     const void* beta,  const void* C,       void* D,
                                     void *workspace, uint64_t workspaceSize, hipStream_t stream)
{

    if (!handle || !A || !B || !D)
	    return HIPTENSOR_STATUS_NOT_INITIALIZED;

    std::cout << "Entered the " << __func__ << std::endl;
    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // device operation
    auto op       = DeviceOpInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument((ADataType *) A,
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

    if(!op.IsSupportedArgument(argument))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return HIPTENSOR_STATUS_SUCCESS;
    }
    

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, true});

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

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " 
	    << op.GetTypeString() << std::endl;
    std::cout << "Exited the " << __func__ << std::endl;

    return HIPTENSOR_STATUS_SUCCESS;
}
