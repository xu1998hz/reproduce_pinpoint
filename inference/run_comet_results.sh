for wmt in "wmt22" "wmt23"
do  
    for lang_dir in "zh-en" "en-de"
    do
        for model_type in "mistral" "llama2"
        do  
            echo "${wmt} ${lang_dir} ${model_type}"
            CUDA_VISIBLE_DEVICES=2 python3 inference/run_comet_eval.py --lang_dir "${lang_dir}" --wmt "${wmt}" --model_type "${model_type}"
        done
    done 
done