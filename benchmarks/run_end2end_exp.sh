
test_id=0








for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    
    for tot_gpu_num in 8
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        for max_group_seq_num in 1 10 20 40
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=machine2


            specify_outlen=
            
            reqnum=10000
            router_question_version=not_multiple_choice_question
            outlen_known=--outlen_known
            
            outlen_known_str=_outlen_known_
            
            for use_specify_outlen in no yes
            do

                if [ $use_specify_outlen = yes ] && [ $outlen_known = --outlen_known ]; then
                    continue
                fi

                specify_outlen=
                outlen_file_name_setting=maxlen_4096
                if [ $use_specify_outlen = yes ]; then
                    specify_outlen=--specify_outlen
                    outlen_file_name_setting=setOutlen
                fi

                echo use_specify_outlen: $use_specify_outlen  specify_outlen: $specify_outlen

                
                for router_replicate_num in 1 2 3 4
                do

                    
                    
                    
                    
                    
                    
                    
                    for method in greedy_saturn_LPT min_gpu_useAll max_gpu
                    do 
                        gen_execplans_baseline=
                        search_method_baseline=
                        if [ $method = ours ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=ours
                        fi

                        if [ $method = greedy_saturn ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn
                        fi

                        if [ $method = greedy_saturn_LPT ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_LPT
                        fi

                        if [ $method = greedy_saturn_LPT_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_LPT_no_preemption
                        fi

                        if [ $method = greedy_saturn_flexible_fuse ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse
                        fi

                        if [ $method = greedy_saturn_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_penalty
                        fi

                        if [ $method = greedy_saturn_flexible_fuse_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_penalty
                        fi           

                        if [ $method = max_gpu ]; then
                            gen_execplans_baseline=max_gpu
                            search_method_baseline=ours
                        fi

                        if [ $method = min_gpu ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu
                        fi

                        if [ $method = min_gpu_useAll ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll
                        fi   


                        if [ $method = min_gpu_useAll_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_penalty
                        fi      

                        if [ $method = greedy_saturn_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_no_preemption
                        fi  

                        if [ $method = greedy_saturn_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_penalty_no_preemption
                        fi  

                        if [ $method = greedy_saturn_flexible_fuse_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_no_preemption
                        fi  

                        if [ $method = greedy_saturn_flexible_fuse_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_penalty_no_preemption
                        fi 

                        if [ $method = min_gpu_useAll_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_no_preemption
                        fi 

                        if [ $method = min_gpu_useAll_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_penalty_no_preemption
                        fi             

                        if [ $max_group_seq_num -gt 1 ] && [ $method != ours ]; then
                            continue
                        fi

                        bugfix=_bugfixV2
                        bugfix=_bugfixV2_modelTP1.05
                        bugfix=_bugfixV2_modelTP1.05_searchonly

                        if [ -a test_end2end_schedule/test_${tot_gpu_num}gpu-router_${method}_${gpu_name}_${machine_name}_${router_question_version}_${outlen_file_name_setting}${outlen_known_str}${reqnum}_${router_replicate_num}_${max_group_seq_num}_${test_id}${bugfix}.log ]; then
                            echo "skip test_end2end_schedule/test_${tot_gpu_num}gpu-router_${method}_${gpu_name}_${machine_name}_${router_question_version}_${outlen_file_name_setting}${outlen_known_str}${reqnum}_${router_replicate_num}_${max_group_seq_num}_${test_id}${bugfix}.log"
                            continue
                        fi

                        echo "CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case router --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --router_replicate_num $router_replicate_num --router_question_version $router_question_version --max_token_num 4096  $specify_outlen $outlen_known --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-router_${method}_${gpu_name}_${machine_name}_${router_question_version}_${outlen_file_name_setting}${outlen_known_str}${reqnum}_${router_replicate_num}_${max_group_seq_num}_${test_id}${bugfix}.log"

                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case router --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --router_replicate_num $router_replicate_num --router_question_version $router_question_version --max_token_num 4096  $specify_outlen $outlen_known --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-router_${method}_${gpu_name}_${machine_name}_${router_question_version}_${outlen_file_name_setting}${outlen_known_str}${reqnum}_${router_replicate_num}_${max_group_seq_num}_${test_id}${bugfix}.log
                    done
                done
            done

            specify_outlen=
            
            for reqnum in 1000 2000 5000 10000
            
            
            do
                for max_token_num in 512 256
                do
                    
                    
                    
                    
                    
                    
                    
                    
                    for method in greedy_saturn_LPT
                    do 
                        gen_execplans_baseline=
                        search_method_baseline=
                        if [ $method = ours ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=ours
                        fi

                        if [ $method = greedy_saturn ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn
                        fi

                        if [ $method = greedy_saturn_LPT ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_LPT
                        fi

                        if [ $method = greedy_saturn_flexible_fuse ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse
                        fi

                        if [ $method = greedy_saturn_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_penalty
                        fi

                        if [ $method = greedy_saturn_flexible_fuse_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_penalty
                        fi


                        if [ $method = max_gpu ]; then
                            gen_execplans_baseline=max_gpu
                            search_method_baseline=ours
                        fi

                        if [ $method = min_gpu ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu
                        fi

                        if [ $method = min_gpu_useAll ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll
                        fi

                        if [ $method = min_gpu_useAll_penalty ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_penalty
                        fi 


                        if [ $method = greedy_saturn_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_no_preemption
                        fi  

                        if [ $method = greedy_saturn_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_penalty_no_preemption
                        fi  

                        if [ $method = greedy_saturn_flexible_fuse_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_no_preemption
                        fi  

                        if [ $method = greedy_saturn_flexible_fuse_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=greedy_saturn_flexible_fuse_penalty_no_preemption
                        fi 

                        if [ $method = min_gpu_useAll_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_no_preemption
                        fi 

                        if [ $method = min_gpu_useAll_penalty_no_preemption ]; then
                            gen_execplans_baseline=ours
                            search_method_baseline=min_gpu_useAll_penalty_no_preemption
                        fi             


                        
                        if [ $max_group_seq_num -gt 1 ] && [ $method != ours ]; then
                            continue
                        fi


                        bugfix=_bugfixV2
                        bugfix=_bugfixV2_modelTP1.05
                        bugfix=_bugfixV2_modelTP1.05_searchonly



                        if [ -a test_end2end_schedule/test_${tot_gpu_num}gpu-llm-blender_${method}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}${bugfix}.log ]; then
                            echo "skip test_end2end_schedule/test_${tot_gpu_num}gpu-llm-blender_${method}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}${bugfix}.log"
                            continue
                        fi
                        echo "CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case general --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-llm-blender_${method}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}${bugfix}.log"

                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case general --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $specify_outlen --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-llm-blender_${method}_${gpu_name}_${machine_name}_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}${bugfix}.log
                    done
                done
            done
        done
    done
done































test_id=0


for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    
    for tot_gpu_num in 8 
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        
        for max_group_seq_num in 1
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=machine2

            specify_outlen=
            
            
            for summarize_model in lmsys/vicuna-13b-v1.5 
            do
                summarize_model_setting=vicuna-13b-v1.5
                if [ $summarize_model = mistralai/Mixtral-8x7B-Instruct-v0.1 ]; then
                    summarize_model_setting=Mixtral-8x7B-Instruct-v0.1
                fi
                
                for reqnum in 200 300 400 100 500
                
                do
                    
                    for evaluator_num in 4
                    do
                        
                        for max_token_num in 900
                        do
                            
                            for reqnum_mixed_blender in 5000
                            do
                                
                                for max_token_num_mixed_blender in 512 256
                                do
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    for method in greedy_saturn_LPT greedy_saturn_LPT_no_preemption
                                    do
                                        gen_execplans_baseline=
                                        search_method_baseline=
                                        if [ $method = ours ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=ours
                                        fi

                                        if [ $method = greedy_saturn ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn
                                        fi

                                        if [ $method = greedy_saturn_LPT ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_LPT
                                        fi     

                                        if [ $method = greedy_saturn_LPT_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_LPT_no_preemption
                                        fi                                   

                                        if [ $method = greedy_saturn_flexible_fuse ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse
                                        fi

                                        if [ $method = greedy_saturn_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_penalty
                                        fi

                                        if [ $method = greedy_saturn_flexible_fuse_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_penalty
                                        fi

                                        if [ $method = greedy_saturn_consider_max_latency ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_consider_max_latency
                                        fi

                                        if [ $method = max_gpu ]; then
                                            gen_execplans_baseline=max_gpu
                                            search_method_baseline=ours
                                        fi

                                        if [ $method = min_gpu ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu
                                        fi

                                        if [ $method = min_gpu_useAll ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll
                                        fi

                                        if [ $method = min_gpu_useAll_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_penalty
                                        fi

                                        if [ $method = greedy_saturn_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_penalty_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_flexible_fuse_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_flexible_fuse_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_penalty_no_preemption
                                        fi 

                                        if [ $method = min_gpu_useAll_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_no_preemption
                                        fi 

                                        if [ $method = min_gpu_useAll_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_penalty_no_preemption
                                        fi  


                                        bugfix=
                                        
                                        
                                        

                                        if [[ "$method" == *"greedy_saturn"* ]] || [[ "$method" == *"min_gpu_useAll"* ]]; then
                                            bugfix=_bugfixV2
                                        fi

                                        
                                        bugfix=_bugfixV2           

                                        
                                        bugfix=_bugfixV3

                                        if [[ "$method" == *"greedy_saturn_LPT"* ]]; then
                                            bugfix=_bugfixV2_modelTP1.05
                                            bugfix=_bugfixV2_modelTP1.05_searchonly
                                        fi                                          

                                        
                                        if [ $max_group_seq_num -gt 1 ] && [ $method != ours ]; then
                                            continue
                                        fi

                                        if [ -a test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log ]; then
                                            echo skip test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                                            continue
                                        fi

                                        echo "CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case mixed --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --reqnum_mixed_blender $reqnum_mixed_blender --max_token_num_mixed_blender $max_token_num_mixed_blender --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log"

                                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case mixed --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --reqnum_mixed_blender $reqnum_mixed_blender --max_token_num_mixed_blender $max_token_num_mixed_blender --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
















test_id=0


for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    
    for tot_gpu_num in 8 
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        
        for max_group_seq_num in 1
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=machine2

            specify_outlen=
            outlen_known=--outlen_known
            outlen_known_str=_outlen_known_
            
            
            for summarize_model in lmsys/vicuna-13b-v1.5 
            do
                summarize_model_setting=vicuna-13b-v1.5
                if [ $summarize_model = mistralai/Mixtral-8x7B-Instruct-v0.1 ]; then
                    summarize_model_setting=Mixtral-8x7B-Instruct-v0.1
                fi
                
                for reqnum in 300 400 100 200 500
                do
                    
                    for evaluator_num in 4
                    do
                        
                        for max_token_num in 900
                        do
                            
                            for reqnum_mixed_blender in 5000
                            do
                                for max_token_num_mixed_blender in 256 512
                                do
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    for method in greedy_saturn_LPT
                                    do
                                        gen_execplans_baseline=
                                        search_method_baseline=
                                        if [ $method = ours ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=ours
                                        fi

                                        if [ $method = greedy_saturn ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn
                                        fi

                                        if [ $method = greedy_saturn_LPT ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_LPT
                                        fi                                        

                                        if [ $method = greedy_saturn_flexible_fuse ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse
                                        fi

                                        if [ $method = greedy_saturn_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_penalty
                                        fi

                                        if [ $method = greedy_saturn_flexible_fuse_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_penalty
                                        fi

                                        if [ $method = greedy_saturn_consider_max_latency ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_consider_max_latency
                                        fi

                                        if [ $method = max_gpu ]; then
                                            gen_execplans_baseline=max_gpu
                                            search_method_baseline=ours
                                        fi

                                        if [ $method = min_gpu ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu
                                        fi

                                        if [ $method = min_gpu_useAll ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll
                                        fi

                                        if [ $method = min_gpu_useAll_penalty ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_penalty
                                        fi

                                        if [ $method = greedy_saturn_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_penalty_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_flexible_fuse_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_no_preemption
                                        fi  

                                        if [ $method = greedy_saturn_flexible_fuse_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=greedy_saturn_flexible_fuse_penalty_no_preemption
                                        fi 

                                        if [ $method = min_gpu_useAll_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_no_preemption
                                        fi 

                                        if [ $method = min_gpu_useAll_penalty_no_preemption ]; then
                                            gen_execplans_baseline=ours
                                            search_method_baseline=min_gpu_useAll_penalty_no_preemption
                                        fi  


                                        bugfix=
                                        
                                        
                                        

                                        if [[ "$method" == *"greedy_saturn"* ]] || [[ "$method" == *"min_gpu_useAll"* ]]; then
                                            bugfix=_bugfixV2
                                        fi

                                        bugfix=_bugfixV2

                                        bugfix=_bugfixV2_modelTP1.05
                                        bugfix=_bugfixV2_modelTP1.05_searchonly                                   

                                        
                                        if [ $max_group_seq_num -gt 1 ] && [ $method != ours ]; then
                                            continue
                                        fi

                                        if [ -a test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}${outlen_known_str}${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log ]; then
                                            echo skip test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}${outlen_known_str}${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                                            continue
                                        fi

                                        echo "CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case mixed --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $outlen_known --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --reqnum_mixed_blender $reqnum_mixed_blender --max_token_num_mixed_blender $max_token_num_mixed_blender --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}${outlen_known_str}${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log"

                                        CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case mixed --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num $outlen_known --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --reqnum_mixed_blender $reqnum_mixed_blender --max_token_num_mixed_blender $max_token_num_mixed_blender --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-mixed_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}${outlen_known_str}${reqnum}_maxlenblender_${max_token_num_mixed_blender}_${reqnum_mixed_blender}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done















test_id=0

for gpu_name in A100-80G
do
    byte_per_gpu=85899345920
    if [ $gpu_name = A100-40G ]; then
        byte_per_gpu=42949672960
    fi
    
    for tot_gpu_num in 8 
    do
        gpu_ids=0,1,2,3,4,5,6,7
        if [ $tot_gpu_num -eq 4 ]; then
            gpu_ids=0,1,2,3
        fi
        
        for max_group_seq_num in 1 10
        do 
            top_k=20
            similar_threshold=0.2
            fully_connected_gpu_unit=2
            machine_name=machine2

            specify_outlen=
            
            
            for summarize_model in lmsys/vicuna-13b-v1.5 
            do
                summarize_model_setting=vicuna-13b-v1.5
                if [ $summarize_model = mistralai/Mixtral-8x7B-Instruct-v0.1 ]; then
                    summarize_model_setting=Mixtral-8x7B-Instruct-v0.1
                fi
                
                for reqnum in 100 200 300 400 500
                do
                    
                    for evaluator_num in 2 4 8
                    do
                        
                        for max_token_num in 100 500 900
                        do

                            if ! { { [ "$evaluator_num" -eq 4 ] && [ "$max_token_num" -eq 900 ]; } || { { [ "$reqnum" -eq 300 ] || [ "$reqnum" -eq 400 ]; } && [ "$max_token_num" -eq 900 ]; } || { { [ "$reqnum" -eq 300 ] || [ "$reqnum" -eq 400 ]; } && [ "$evaluator_num" -eq 4 ]; } }; then
                                continue
                            fi


                            
                            
                            
                            
                            
                            
                            
                            
                            for method in greedy_saturn_LPT
                            do
                                gen_execplans_baseline=
                                search_method_baseline=
                                if [ $method = ours ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=ours
                                fi

                                if [ $method = greedy_saturn ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn
                                fi

                                if [ $method = greedy_saturn_LPT ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_LPT
                                fi


                                if [ $method = greedy_saturn_flexible_fuse ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_flexible_fuse
                                fi

                                if [ $method = greedy_saturn_penalty ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_penalty
                                fi

                                if [ $method = greedy_saturn_flexible_fuse_penalty ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_flexible_fuse_penalty
                                fi

                                if [ $method = max_gpu ]; then
                                    gen_execplans_baseline=max_gpu
                                    search_method_baseline=ours
                                fi

                                if [ $method = min_gpu ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=min_gpu
                                fi

                                if [ $method = min_gpu_useAll ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=min_gpu_useAll
                                fi

                                if [ $method = min_gpu_useAll_penalty ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=min_gpu_useAll_penalty
                                fi

                                if [ $method = greedy_saturn_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_no_preemption
                                fi  

                                if [ $method = greedy_saturn_penalty_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_penalty_no_preemption
                                fi  

                                if [ $method = greedy_saturn_flexible_fuse_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_flexible_fuse_no_preemption
                                fi  

                                if [ $method = greedy_saturn_flexible_fuse_penalty_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=greedy_saturn_flexible_fuse_penalty_no_preemption
                                fi 

                                if [ $method = min_gpu_useAll_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=min_gpu_useAll_no_preemption
                                fi 

                                if [ $method = min_gpu_useAll_penalty_no_preemption ]; then
                                    gen_execplans_baseline=ours
                                    search_method_baseline=min_gpu_useAll_penalty_no_preemption
                                fi  

                                bugfix=
                                
                                
                                

                                if [[ "$method" == *"greedy_saturn"* ]] || [[ "$method" == *"min_gpu_useAll"* ]]; then
                                    bugfix=_bugfixV2
                                fi


                                
                                bugfix=_bugfixV2
                                bugfix=_bugfixV2_modelTP1.05   

                                
                                if [ $max_group_seq_num -gt 1 ] && [ $method != ours ]; then
                                    continue
                                fi

                                if [ -a test_end2end_schedule/test_${tot_gpu_num}gpu-booookscore_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log ]; then
                                    echo skip test_${tot_gpu_num}gpu-booookscore_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                                    continue
                                fi

                                echo "CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case chain-summary --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-booookscore_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log"

                                CUDA_VISIBLE_DEVICES=$gpu_ids python3 schedule_multi_model.py --gen-execplans-baseline $gen_execplans_baseline --search_method_baseline $search_method_baseline --test-case chain-summary --ratio-seed 0 --ratio-set 1 --reqnum $reqnum --max_token_num $max_token_num --gpu_name $gpu_name --byte_per_gpu $byte_per_gpu --tot_gpu_num $tot_gpu_num --max_group_seq_num $max_group_seq_num --top_k $top_k --similar_threshold $similar_threshold --fully_connected_gpu_unit $fully_connected_gpu_unit --machine_name $machine_name --evaluator_num $evaluator_num --summarize_model $summarize_model --evaluator_model meta-llama/Llama-2-70b-chat-hf --test_id $test_id >> test_end2end_schedule/test_${tot_gpu_num}gpu-booookscore_${method}_${gpu_name}_${machine_name}_${summarize_model_setting}_${evaluator_num}eval_maxlen_${max_token_num}_${reqnum}_${max_group_seq_num}_${test_id}_expOutlen${bugfix}.log
                            done
                        done
                    done
                done
            done
        done
    done
done





