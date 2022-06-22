#include <iostream>
#include <fstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "check_err.hpp"
#include "config.hpp"
#include "ht_tensor.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_contraction_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"
#include "ht_tensor_generator_utility.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

typedef float ADataType;
typedef float BDataType;
typedef float CDataType;
using AccDataType = float;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

//static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,              4>;
// clang-format on



// hardcoded for NumDimM == NumDimN == NumDimK == 2
template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2, bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ks_ns,
                 Tensor<CDataType>& c_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ks_ns_{b_ks_ns},
              c_ms_ns_{c_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ks_ns_;
        Tensor<CDataType>& c_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                const int K0 = arg.a_ms_ks_.mDesc.GetLengths()[2];
                const int K1 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    for(int k1 = 0; k1 < K1; ++k1)
                    {
                        AccDataType v_a;
                        AccDataType v_b;

                        arg.a_element_op_(
                            v_a, static_cast<const AccDataType>(arg.a_ms_ks_(m0, m1, k0, k1)));
                        arg.b_element_op_(
                            v_b, static_cast<const AccDataType>(arg.b_ks_ns_(k0, k1, n0, n1)));

                        v_acc += v_a * v_b;
                    }
                }

                AccDataType v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_ms_ns_(m0, m1, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.c_ms_ns_.mDesc.GetLengths()[0],
                                       arg.c_ms_ns_.mDesc.GetLengths()[1],
                                       arg.c_ms_ns_.mDesc.GetLengths()[2],
                                       arg.c_ms_ns_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const ck::tensor_operation::device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const ck::tensor_operation::device::BaseArgument*) override
    {
        return true;
    }

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ks_ns,
                             Tensor<CDataType>& c_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_ms_ks, b_ks_ns, c_ms_ns, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<ck::tensor_operation::device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceContraction_M2_N2_K2"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

using ReferenceOpInstance = ReferenceContraction_M2_N2_K2<NumDimM,
                                                          NumDimN,
                                                          NumDimK,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 3;
    bool time_kernel     = false;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value, 3=cutensor_style_init)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    std::ofstream tensorA, tensorB, tensorC;

    typedef float floatTypeCompute;

    hiptensorDataType_t typeA = HIPTENSOR_R_32F;
    hiptensorDataType_t typeB = HIPTENSOR_R_32F;
    hiptensorDataType_t typeC = HIPTENSOR_R_32F;
    hiptensorComputeType_t typeCompute = HIPTENSOR_COMPUTE_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

    std::cout << "RAND_MAX value is " << RAND_MAX << std::endl;

    /**********************
     * Computing: C_{m,n,u,v} = A_{m,n,h,k} B_{h,k,u,v}
     **********************/

    std::vector<int> modeC{'m','n','u','v'};
    std::vector<int> modeA{'m','n','h','k'};
    std::vector<int> modeB{'h','k','u','v'};


    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    
    extent['m'] = 5;
    extent['n'] = 6;
    extent['u'] = 3;
    extent['v'] = 4;
    extent['h'] = 3;
    extent['k'] = 4;
    
    std::vector<int64_t> c_ms_ns_lengths;
    for (auto mode : modeC)
    	c_ms_ns_lengths.push_back(extent[mode]);
    std::vector<int64_t> a_ms_ks_lengths;
    for (auto mode : modeA)
       	a_ms_ks_lengths.push_back(extent[mode]);
    std::vector<int64_t> b_ks_ns_lengths;
    for (auto mode : modeB)
        b_ks_ns_lengths.push_back(extent[mode]);


    hiptensorHandle_t handle;
    hiptensorInit(&handle);
    
    /********************************************
     * Intialise Tensors with the input lengths *
     ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks;
    std::cout << "a_ms_ks: ";
    hiptensorInitTensorDescriptor(&handle, &a_ms_ks, nmodeA, 
				a_ms_ks_lengths.data(), NULL,/*stride*/
				typeA, HIPTENSOR_OP_IDENTITY);

    hiptensorTensorDescriptor_t b_ks_ns;
    std::cout << "b_ks_ns: ";
    hiptensorInitTensorDescriptor(&handle, &b_ks_ns, nmodeB,
                       		b_ks_ns_lengths.data(), NULL,/*stride*/
				typeB, HIPTENSOR_OP_IDENTITY);
    
    hiptensorTensorDescriptor_t c_ms_ns;
    std::cout << "c_ms_ns: ";
    hiptensorInitTensorDescriptor(&handle, 
				&c_ms_ns, nmodeC,
				c_ms_ns_lengths.data(), NULL,/*stride*/
                      		typeC, HIPTENSOR_OP_IDENTITY);


    size_t sizeA = sizeof(ADataType) * a_ms_ks.hiptensorGetElementSpace();
    size_t sizeB = sizeof(BDataType) * b_ks_ns.hiptensorGetElementSpace();
    size_t sizeC = sizeof(CDataType) * c_ms_ns.hiptensorGetElementSpace();

    void *A_d, *B_d, *C_d;
    hip_check_error(hipMalloc(static_cast<void**>(&A_d), sizeA));
    hip_check_error(hipMalloc(static_cast<void**>(&B_d), sizeB));
    hip_check_error(hipMalloc(static_cast<void**>(&C_d), sizeC));


    ADataType *A = (ADataType*) malloc(sizeA);
    BDataType *B = (BDataType*) malloc(sizeB);
    CDataType *C = (CDataType*) malloc(sizeC);
	
    /********************************************
     * Transfer the Host Tensor to Device Memory *
     ********************************************/
    hip_check_error(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    hip_check_error(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

    /************************************************
     * Retrieve the memory alignment for each tensor
     ************************************************/ 
    uint32_t alignmentRequirementA;
    hiptensorGetAlignmentRequirement(&handle,
                          A_d, &a_ms_ks,
                          &alignmentRequirementA);
    std::cout << "Tensor A element space: " << alignmentRequirementA << std::endl;

    uint32_t alignmentRequirementB;
    hiptensorGetAlignmentRequirement(&handle,
                          B_d, &b_ks_ns,
                          &alignmentRequirementB);
    std::cout << "Tensor B element space: " << alignmentRequirementB << std::endl;

    uint32_t alignmentRequirementC;
    hiptensorGetAlignmentRequirement(&handle,
                          C_d, &c_ms_ns,
                          &alignmentRequirementC);
    std::cout << "Tensor C element space: " << alignmentRequirementC << std::endl;;

#if 0
    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // device operation
    auto op       = DeviceOpInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(static_cast<ADataType*>(a_ms_ks.ht_devmem.GetDeviceBuffer()),
                                    static_cast<BDataType*>(b_ks_ns.ht_devmem.GetDeviceBuffer()),
                                    static_cast<CDataType*>(c_ms_ns.ht_devmem.GetDeviceBuffer()),
                                    a_ms_ks_lengths,
                                    std::vector<ck::index_t>(a_ms_ks.ht_tensor.mDesc.GetStrides().begin(), a_ms_ks.ht_tensor.mDesc.GetStrides().end()),
                                    b_ks_ns_lengths,
				    std::vector<ck::index_t>(b_ks_ns.ht_tensor.mDesc.GetStrides().begin(), b_ks_ns.ht_tensor.mDesc.GetStrides().end()),
                                    c_ms_ns_lengths,
                                    std::vector<ck::index_t>(c_ms_ns.ht_tensor.mDesc.GetStrides().begin(), c_ms_ns.ht_tensor.mDesc.GetStrides().end()),
                                    a_element_op,
                                    b_element_op,
                                    c_element_op);

    if(!op.IsSupportedArgument(argument))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    ck::index_t M = std::accumulate(c_ms_ns_lengths.begin(),
                                    c_ms_ns_lengths.begin() + NumDimM,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t N = std::accumulate(c_ms_ns_lengths.begin() + NumDimM,
                                    c_ms_ns_lengths.begin() + NumDimM + NumDimN,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t K = std::accumulate(a_ms_ks_lengths.begin() + NumDimM,
                                    a_ms_ks_lengths.begin() + NumDimM + NumDimK,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    //c_ms_ns_device_buf.FromDevice(c_ms_ns_device_result.mData.data());

    tensorA.open("tensor_A.txt");
    LogRangeToFile<ADataType>(tensorA, a_ms_ks.ht_tensor.mData, ","); 
    LogRangeAsType<ADataType>(std::cout<<"Tensor A elements:\n", a_ms_ks.ht_tensor.mData,",");
    std::cout<<std::endl;
    tensorA.close();
    tensorB.open("tensor_B.txt");
    LogRangeToFile<BDataType>(tensorB, b_ks_ns.ht_tensor.mData, ","); 
    LogRangeAsType<BDataType>(std::cout<<"Tensor B elements:\n", b_ks_ns.ht_tensor.mData,",");
    std::cout<<std::endl;
    tensorB.close();
#if 0
    if(do_verification)
    {
        auto ref_gemm    = ReferenceOpInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_ms_ks, b_ks_ns, c_ms_ns_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        return ck::utils::check_err(c_ms_ns_device_result.mData, c_ms_ns_host_result.mData) ? 0 : 1;
    }
#endif
    tensorC.open("tensor_C_contraction_results.txt");
    LogRangeToFile<CDataType>(tensorC, c_ms_ns.ht_tensor.mData, ","); 
    LogRangeAsType<CDataType>(std::cout<<"Tensor C elements:\n", c_ms_ns.ht_tensor.mData, ","); 
    std::cout<<std::endl;
    tensorC.close();
#endif
	
    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) hip_check_error(hipFree(A_d));
    if (B_d) hip_check_error(hipFree(B_d));
    if (C_d) hip_check_error(hipFree(C_d));
    
    return 0;
}
