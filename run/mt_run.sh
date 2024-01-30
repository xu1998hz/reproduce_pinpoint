# ./main_run.sh -i data/mt_test_data/en-de_wmt_test_wmt22.tsv -b 1 -m mistral -s 7 -p fp16 -o out/ -t mt -d False

for wmt in "wmt22" "wmt23"
do  
    for lang_dir in "zh-en" "en-de"
    do
        for model_type in "mistral" "llama2"
        do
            echo "${wmt} ${lang_dir} ${model_type}"
            ./main_run.sh -i data/mt_test_data/${lang_dir}_wmt_test_${wmt}.tsv -b 1 -m ${model_type} -s 7 -p fp16 -o out/ -t mt -d False
        done
    done
done