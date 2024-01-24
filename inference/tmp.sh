python3 postprocess.py --data_file /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/zh-en_wmt_test_wmt22_out_llama2.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt22_out_clean_llama2.pkl --model_base llama --language zh-en --task mt
python3 comet_eval.py --data /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt22_out_clean_llama2.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/mt_out/comet_scores_zh-en_wmt_test_wmt22_llama2.json 

python3 postprocess.py --data_file /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/zh-en_wmt_test_wmt23_out_llama2.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt23_out_clean_llama2.pkl --model_base llama --language zh-en --task mt
python3 comet_eval.py --data /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt23_out_clean_llama2.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/mt_out/comet_scores_zh-en_wmt_test_wmt23_llama2.json 

python3 postprocess.py --data_file /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/zh-en_wmt_test_wmt22_out_mistral.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt22_out_clean_mistral.pkl --model_base mistral --language zh-en --task mt
python3 comet_eval.py --data /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt22_out_clean_mistral.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/mt_out/comet_scores_zh-en_wmt_test_wmt22_mistral.json 

python3 postprocess.py --data_file /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/zh-en_wmt_test_wmt23_out_mistral.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt23_out_clean_mistral.pkl --model_base mistral --language zh-en --task mt
python3 comet_eval.py --data /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/data/zh-en_wmt_test_wmt23_out_clean_mistral.pkl --out_path /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/out/mt_out/comet_scores_zh-en_wmt_test_wmt23_mistral.json 

