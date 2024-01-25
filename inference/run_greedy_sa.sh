# wmt="wmt23"
# model_type="mistral"
# CUDA_VISIBLE_DEVICES=3 python3 inference/greedy_sa.py --wmt "${wmt}" --lang zh-en \
# --model "${model_type}" --model_addr /mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/ \
# --data_path "out/mt_out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.json" \
# --out_path "out/mt_out/correction_zh-en_wmt_test_${wmt}_${model_type}.json" \
# --batch_size 1 --max_length 720 --feedback_addr "out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.txt" \
# --feedback_type "improve"

for wmt in "wmt22" "wmt23"
do
    for model_type in "mistral" "llama2"
    do 
        feedback_type="improve"
        CUDA_VISIBLE_DEVICES=0 nohup python3 inference/greedy_sa.py --wmt "${wmt}" --lang zh-en \
        --model "${model_type}" --model_addr /mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/ \
        --data_path "out/mt_out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.json" \
        --out_path "out/mt_out/correction_zh-en_wmt_test_${wmt}_${model_type}_${feedback_type}.json" \
        --batch_size 1 --max_length 720 --feedback_addr "out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.txt" \
        --feedback_type "${feedback_type}" > "${wmt}_${model_type}_${feedback_type}.out" 2>&1 &

        feedback_type="binary"
        CUDA_VISIBLE_DEVICES=1 nohup python3 inference/greedy_sa.py --wmt "${wmt}" --lang zh-en \
        --model "${model_type}" --model_addr /mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/ \
        --data_path "out/mt_out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.json" \
        --out_path "out/mt_out/correction_zh-en_wmt_test_${wmt}_${model_type}_${feedback_type}.json" \
        --batch_size 1 --max_length 720 --feedback_addr "out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.txt" \
        --feedback_type "${feedback_type}" > "${wmt}_${model_type}_${feedback_type}.out" 2>&1 &

        feedback_type="score"
        CUDA_VISIBLE_DEVICES=2 nohup python3 inference/greedy_sa.py --wmt "${wmt}" --lang zh-en \
        --model "${model_type}" --model_addr /mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/ \
        --data_path "out/mt_out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.json" \
        --out_path "out/mt_out/correction_zh-en_wmt_test_${wmt}_${model_type}_${feedback_type}.json" \
        --batch_size 1 --max_length 720 --feedback_addr "out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.txt" \
        --feedback_type "${feedback_type}" > "${wmt}_${model_type}_${feedback_type}.out" 2>&1 &

        feedback_type="mqm"
        CUDA_VISIBLE_DEVICES=3 nohup python3 inference/greedy_sa.py --wmt "${wmt}" --lang zh-en \
        --model "${model_type}" --model_addr /mnt/taurus/home/guangleizhu/reproduce_pinpoint/finetune/ft_out/zh-en/checkpoint-760/ \
        --data_path "out/mt_out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.json" \
        --out_path "out/mt_out/correction_zh-en_wmt_test_${wmt}_${model_type}_${feedback_type}.json" \
        --batch_size 1 --max_length 720 --feedback_addr "out/comet_scores_zh-en_wmt_test_${wmt}_${model_type}.txt" \
        --feedback_type "${feedback_type}" > "${wmt}_${model_type}_${feedback_type}.out" 2>&1 &
        
        echo "${wmt} and ${model_type}"
        echo "all four processes start to run!"
        wait
        echo "all four processes finished!"
    done
done