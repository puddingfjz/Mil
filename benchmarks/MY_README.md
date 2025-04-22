# code structure


    benchmark_throughput.py:                end-to-end single model 

    output_length_sampler.py:               sampling output length
    
    fake_scheduling.py:                     doing fake scheduling to estimate cost
    

    construct_cost_model.py:                run this file to collect data to build cost model
    
    my_per_iter_latency_estimator.py:       about the cost model: linear functions, etc.
    
    model_coeff_database.py:                stores the coefficients for each model to compute the flops
                                            obtained by running comp_model_size.py

    comp_model_size.py:                     run this file (1) to get the model parameter sizes and store them in
                                            model_size_database.py
                                            and (2) to get the model coefficients to compute flops and store them in model_coeff_database.py

    model_size_database.py:                 stores the model parameter sizes
                                            obtained by running comp_model_size.py

    model_initcost_database.py              stores the model init costs
                                            obtained by running comp_model_size.py
                                            NOTE: every time introducing a new model (i.e., the models except those used in LLM-Blender), run comp_model_size.py 
                                                for that model.

    search_exec_plans.py:                   generate multi-model schedule plan
                                            


    schedule_multi_model.py:                schedule the end-to-end inference

    my_llm_infer_worker_multiprocessing.py  inference code for each data parallel worker.
                                            will be called by bench_throughput.py run_vllm.
    

    multimodel_scheduler.py                 the communication betwen multiple model processes.



