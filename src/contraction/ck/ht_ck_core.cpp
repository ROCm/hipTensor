#include <iomanip>
#include <numeric>
#include <vector>
#include <iostream>
#include "ck.hpp"
#include "tensor_layout.hpp"
#include "device_contraction_multiple_d.hpp"
#include "element_wise_operation.hpp"
#include "contraction_scale.hpp"
#include "contraction_bilinear.hpp"

#include "ht_types.hpp"
#include "ht_ck_core.hpp"

using F32   		   = float;
using ADataType        = F32;
using BDataType        = F32;
using CDataType        = F32;
using DDataType        = F32;


static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEScaleElementOp = Scale;
using CDEBilinearElementOp = Bilinear;


using ContractionScaleOp   = ck::tensor_operation::device::DeviceContractionMultipleD<
								NumDimM,
								NumDimN,
								NumDimK,
								ADataType,
								BDataType,
								ck::Tuple<>,
								DDataType,
								ck::tensor_operation::element_wise::PassThrough,
								ck::tensor_operation::element_wise::PassThrough,
								ck::tensor_operation::element_wise::Scale>;

using ContractionBilinearOp = ck::tensor_operation::device::DeviceContractionMultipleD<
        						NumDimM,
        						NumDimN,
        						NumDimK,
        						ADataType,
        						BDataType,
								ck::Tuple<CDataType>,
								DDataType,
								ck::tensor_operation::element_wise::PassThrough,
								ck::tensor_operation::element_wise::PassThrough,
								ck::tensor_operation::element_wise::Bilinear>;

#if 0
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

    for(auto it = desc->ht_contract_attr_desc.begin(); it < desc->ht_contract_attr_desc.end(); ++it)
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

hiptensorStatus_t hiptensorInitCKContractionPlan(const hiptensorHandle_t* handle,
                                            hiptensorContractionPlan_t *plan,
                                            const hiptensorContractionDescriptor_t* desc,
                                            const uint64_t workspaceSize)

{
    if (!handle || !plan)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

    plan->ht_plan_desc = *desc;
    
	const auto op_scale_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionScaleOp>::GetInstances(); 
    std::cout << "found " << op_scale_ptrs.size() << " instances" << std::endl;
   	hiptensorCKContraction(handle, plan, ht_contract_metrics, HIPTENSOR_CONTRACTION_SCALE);
   
	const auto op_bilinear_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionBilinearOp>::GetInstances();                      
    std::cout << "found " << op_bilinear_ptrs.size() << " instances" << std::endl;
    hiptensorCKContraction(handle, plan, ht_contract_metrics, HIPTENSOR_CONTRACTION_BILINEAR);

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptesnorConstructCKContractionArgument( const std::unique_ptr<ContractionBilinearOp>& bilinearOp,
												const std::unique_ptr<ContractionScaleOp>& scaleOp,
										        const hiptensorContractionPlan_t* plan,
										        const void* alpha, const void* A, const void* B,
                              			        const void* beta, const void* C,	void* D,
                                                std::unique_ptr<ck::tensor_operation::device::BaseArgument>& ckContractionArgPtr,
										        const hiptesnorContractionOperation_t contractionOp )
{
	const auto a_element_op   = AElementOp{};
    const auto b_element_op   = BElementOp{};

	if ( !bilinearOp && !scaleOp )
		return  HIPTENSOR_STATUS_NOT_INITIALIZED;
	
    if( contractionOp == HIPTENSOR_CONTRACTION_BILINEAR)
	{
        const auto cde_element_op = CDEBilinearElementOp{ *(F32 *)alpha, *(F32 *)beta };
       	ckContractionArgPtr  = bilinearOp->MakeArgumentPointer(
                       A,
                       B,
                       std::array<const void*, 1>{C},
                       D,
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
                       std::array<std::vector<ck::index_t>, 1>
                                 {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end())},
                       std::array<std::vector<ck::index_t>, 1>
                                 {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end())},
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
                       std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
                       a_element_op,
                       b_element_op,
                       cde_element_op);
    }
    else if (contractionOp == HIPTENSOR_CONTRACTION_SCALE) 
    {   
		const auto cde_element_op = CDEScaleElementOp{ *(F32 *)alpha };
		ckContractionArgPtr = scaleOp->MakeArgumentPointer(
							A,
							B,
							std::array<const void*, 0>{},
							D,
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
							std::array<std::vector<ck::index_t>, 0>{},
							std::array<std::vector<ck::index_t>, 0>{},
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
							a_element_op,
							b_element_op,
							cde_element_op);
    }   
    else
    {
    	std::cout << "Invalid input Contraction operation" << std::endl;
        return HIPTENSOR_STATUS_CK_ERROR;
    }
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

	std::string best_op_name;
    bool found            = false;
	int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    
    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));
	
    auto contraction = [&] (auto &op_layout)
    {
        if (!op_layout)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;

        using ContractionInstance = decltype(op_layout);
        ContractionInstance op = std::move(op_layout);

        const auto a_element_op   = AElementOp{};
        const auto b_element_op   = BElementOp{};
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;

		if( plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_BILINEAR) 
		{
            const auto cde_element_op = CDEBilinearElementOp{ *(F32 *)alpha, *(F32 *)beta };
            argument_ptr  = op->MakeArgumentPointer(
                           A,
                           B,
                           std::array<const void*, 1>{C},
                           D,
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
                           std::array<std::vector<ck::index_t>, 1>
                                     {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end())},
                           std::array<std::vector<ck::index_t>, 1>
                                     {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end())},
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
                           a_element_op,
                           b_element_op,
                           cde_element_op);

		}
		else if ( plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_SCALE )
		{
			const auto cde_element_op = CDEScaleElementOp{ *(F32 *)alpha };
			argument_ptr = op->MakeArgumentPointer(
								A,
								B,
								std::array<const void*, 0>{},
								D,
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
								std::array<std::vector<ck::index_t>, 0>{},
								std::array<std::vector<ck::index_t>, 0>{},
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
								std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
								a_element_op,
								b_element_op,
								cde_element_op);
		}
		else 
		{
			std::cout << "Invalid Contraction operation not supported by CK";
			return HIPTENSOR_STATUS_CK_ERROR;
		}
		
		auto invoker_ptr = op->MakeInvokerPointer();
        std::string op_name = op->GetTypeString();

        if(!op->IsSupportedArgument(argument_ptr.get()))
        {
			std::cout << op->GetTypeString() << " does not support this problem" << std::endl;
            return HIPTENSOR_STATUS_CK_ERROR;
       	}    

		ht_contract_metrics->avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
		hiptensorFillCKContractionMetrics( plan, ht_contract_metrics, plan->ht_plan_desc.ht_contract_op );
		return HIPTENSOR_STATUS_SUCCESS;
	};

	if ( plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_BILINEAR ) 
	{
		const auto op_bilinear_ptrs =  ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionBilinearOp>::GetInstances();
			
		std::cout << "Run all instances and do timing" << std::endl;
		
		for(int i = 0; i < op_bilinear_ptrs.size(); ++i)
    	{
        	//auto& op_ptr = op_bilinear_ptrs[i];
			contraction( op_bilinear_ptrs[i] );
			if(ht_contract_metrics->tflops > best_tflops)
        	{
            	found           = true;
            	best_op_id      = i;
            	best_op_name    = op_bilinear_ptrs[i]->GetTypeString();
            	best_tflops     = ht_contract_metrics->tflops;
            	best_ave_time   = ht_contract_metrics->avg_time;
            	best_gb_per_sec = ht_contract_metrics->transfer_speed;
        	}

		}
		std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    	auto& contract_op_ptr = op_bilinear_ptrs[best_op_id];
    	contraction(contract_op_ptr);
	}
	else if ( plan->ht_plan_desc.ht_contract_op == HIPTENSOR_CONTRACTION_SCALE ) 
	{
		const auto op_scale_ptrs =  ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionScaleOp>::GetInstances();
		
		std::cout << "Run all instances and do timing" << std::endl;
		
		for(int i = 0; i < op_scale_ptrs.size(); ++i)
    	{
        	auto& op_ptr = op_scale_ptrs[i];
			contraction( op_ptr );
			if(ht_contract_metrics->tflops > best_tflops)
        	{
            	found           = true;
            	best_op_id      = i;
            	best_op_name    = op_ptr->GetTypeString();
            	best_tflops     = ht_contract_metrics->tflops;
            	best_ave_time   = ht_contract_metrics->avg_time;
            	best_gb_per_sec = ht_contract_metrics->transfer_speed;
        	}
		}
		
		std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    	auto& contract_op_ptr = op_scale_ptrs[best_op_id];
    	contraction(contract_op_ptr);
	}
	else
	{
		std::cout << "Contraction operation not permitted" << std::endl;
		return HIPTENSOR_STATUS_CK_ERROR;
	}	
	
	return HIPTENSOR_STATUS_SUCCESS;
}
#endif

hiptensorStatus_t hiptensorFillCKContractionMetrics( const hiptensorContractionPlan_t* plan, 
												hiptensorContractionMetrics_t *ht_contract_metrics, 
												const hiptesnorContractionOperation_t contractionOp )
{
	ck::index_t M = std::accumulate(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(),
									plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM,
									ck::index_t{1},
									std::multiplies<ck::index_t>{});

	ck::index_t N = std::accumulate(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM,
									plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin() + NumDimM + NumDimN,
									ck::index_t{1},
								   std::multiplies<ck::index_t>{});

	ck::index_t K= std::accumulate(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin() + NumDimM,
									plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin() + NumDimM + NumDimK,
									ck::index_t{1},
									std::multiplies<ck::index_t>{});

	std::size_t flop = std::size_t(2) * M * N * K;
	std::size_t num_btype;

	if ( contractionOp == HIPTENSOR_CONTRACTION_BILINEAR )
	{
		num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
							sizeof(CDataType) * M * N + sizeof(DDataType) * M * N;
	}
	else if ( contractionOp == HIPTENSOR_CONTRACTION_SCALE)
	{
		num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(DDataType) * M * N;
	}
	else
   	{
   		std::cout << "Input Contraction operation not supported by CK" << std::endl;
     	return HIPTENSOR_STATUS_CK_ERROR;
   	}
	
	ht_contract_metrics->tflops = static_cast<float>(flop) / 1.E9 / ht_contract_metrics->avg_time;
	ht_contract_metrics->transfer_speed = num_btype / 1.E6 / ht_contract_metrics->avg_time;
	return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorCKScaleContraction(const hiptensorHandle_t* handle,
									const hiptensorContractionPlan_t* plan,
									hiptensorContractionMetrics_t *ht_contract_metrics,
									const void* alpha, const void* A, const void* B,
									const void* beta,  const void* C,       void* D,
									void *workspace, uint64_t workspaceSize, hipStream_t stream)
{
	if (!handle || !ht_contract_metrics || !A || !B || !D)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

	std::string best_op_name;
    bool found            = false;
	int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    
    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));
	
    auto contraction_scale = [&] (auto &op_layout)
    {
        if (!op_layout)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;

        using ContractionInstance = decltype(op_layout);
        ContractionInstance op = std::move(op_layout);

        const auto a_element_op   = AElementOp{};
        const auto b_element_op   = BElementOp{};
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;

		const auto cde_element_op = CDEScaleElementOp{ *(F32 *)alpha };
		argument_ptr = op->MakeArgumentPointer(
							A,
							B,
							std::array<const void*, 0>{},
							D,
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
							std::array<std::vector<ck::index_t>, 0>{},
							std::array<std::vector<ck::index_t>, 0>{},
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
							std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
							a_element_op,
							b_element_op,
							cde_element_op);
		
		auto invoker_ptr = op->MakeInvokerPointer();
        std::string op_name = op->GetTypeString();

        if(!op->IsSupportedArgument(argument_ptr.get()))
        {
			std::cout << op->GetTypeString() << " does not support this problem" << std::endl;
            return HIPTENSOR_STATUS_CK_ERROR;
       	}    

		ht_contract_metrics->avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
		hiptensorFillCKContractionMetrics( plan, ht_contract_metrics, plan->ht_plan_desc.ht_contract_op );
		return HIPTENSOR_STATUS_SUCCESS;
	};

	const auto op_scale_ptrs =  ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionScaleOp>::GetInstances();
	
	std::cout << "Run all instances and do timing" << std::endl;
	
	for(int i = 0; i < op_scale_ptrs.size(); ++i)
	{
		auto& op_ptr = op_scale_ptrs[i];
		contraction_scale( op_ptr );
		if(ht_contract_metrics->tflops > best_tflops)
		{
			found           = true;
			best_op_id      = i;
			best_op_name    = op_ptr->GetTypeString();
			best_tflops     = ht_contract_metrics->tflops;
			best_ave_time   = ht_contract_metrics->avg_time;
			best_gb_per_sec = ht_contract_metrics->transfer_speed;
		}
	}
	
	std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
		  << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

	auto& contract_op_ptr = op_scale_ptrs[best_op_id];
	contraction_scale(contract_op_ptr);
	return HIPTENSOR_STATUS_SUCCESS;
}
hiptensorStatus_t hiptensorCKBilinearContraction(const hiptensorHandle_t* handle,
									const hiptensorContractionPlan_t* plan,
									hiptensorContractionMetrics_t *ht_contract_metrics,
									const void* alpha, const void* A, const void* B,
									const void* beta,  const void* C,       void* D,
									void *workspace, uint64_t workspaceSize, hipStream_t stream)
{
	if (!handle || !ht_contract_metrics || !A || !B || !D)
        return HIPTENSOR_STATUS_NOT_INITIALIZED;

	std::string best_op_name;
    bool found            = false;
	int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    
    memset(ht_contract_metrics, 0, sizeof(hiptensorContractionMetrics_t));
	
    auto contraction_bilinear = [&] (auto &op_layout)
    {
        if (!op_layout)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;

        using ContractionInstance = decltype(op_layout);
        ContractionInstance op = std::move(op_layout);

        const auto a_element_op   = AElementOp{};
        const auto b_element_op   = BElementOp{};
        std::unique_ptr<ck::tensor_operation::device::BaseArgument> argument_ptr;

      	const auto cde_element_op = CDEBilinearElementOp{ *(F32 *)alpha, *(F32 *)beta };
        argument_ptr  = op->MakeArgumentPointer(
                           A,
                           B,
                           std::array<const void*, 1>{C},
                           D,
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[0].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[0].strides.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[1].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[1].strides.end()),
                           std::array<std::vector<ck::index_t>, 1>
                                     {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end())},
                           std::array<std::vector<ck::index_t>, 1>
                                     {std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end())},
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].lens.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].lens.end()),
                           std::vector<ck::index_t>(plan->ht_plan_desc.ht_contract_attr_desc[2].strides.begin(), plan->ht_plan_desc.ht_contract_attr_desc[2].strides.end()),
                           a_element_op,
                           b_element_op,
                           cde_element_op);

		auto invoker_ptr = op->MakeInvokerPointer();
        std::string op_name = op->GetTypeString();

        if(!op->IsSupportedArgument(argument_ptr.get()))
        {
			std::cout << op->GetTypeString() << " does not support this problem" << std::endl;
            return HIPTENSOR_STATUS_CK_ERROR;
       	}    

		ht_contract_metrics->avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});
		hiptensorFillCKContractionMetrics( plan, ht_contract_metrics, plan->ht_plan_desc.ht_contract_op );
		return HIPTENSOR_STATUS_SUCCESS;
	};

	const auto op_bilinear_ptrs =  ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<ContractionBilinearOp>::GetInstances();
		
	std::cout << "Run all instances and do timing" << std::endl;
	
	for(int i = 0; i < op_bilinear_ptrs.size(); ++i)
	{
		//auto& op_ptr = op_bilinear_ptrs[i];
		contraction_bilinear( op_bilinear_ptrs[i] );
		if(ht_contract_metrics->tflops > best_tflops)
		{
			found           = true;
			best_op_id      = i;
			best_op_name    = op_bilinear_ptrs[i]->GetTypeString();
			best_tflops     = ht_contract_metrics->tflops;
			best_ave_time   = ht_contract_metrics->avg_time;
			best_gb_per_sec = ht_contract_metrics->transfer_speed;
		}

	}
	std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
		  << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

	auto& contract_op_ptr = op_bilinear_ptrs[best_op_id];
	contraction_bilinear(contract_op_ptr);
	return HIPTENSOR_STATUS_SUCCESS;
}

