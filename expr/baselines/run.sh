run_model(){
    model_name=$1
    tl_strategy=$2
    mkdir -p results/$model_name
    nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > results/$model_name/memory_$tl_strategy.log &
    pid=$!
    python run_baseline.py --model $model_name --tl_strategy $tl_strategy --epochs 25 --output_dir results/$model_name/tl_$tl_strategy --batch_size 8
    kill $pid
}


run_model "mobilenet_v2" "1"
run_model "mobilenet_v2" "2"
run_model "mobilenet_v2" "3"
run_model "resnet18" "1"
run_model "resnet18" "2"
run_model "resnet18" "3"
run_model "resnet34" "1"
run_model "resnet34" "2"
run_model "resnet34" "3"
run_model "resnet50" "1"
run_model "resnet50" "2"
run_model "resnet50" "3"
run_model "vgg16" "1"
run_model "vgg16" "2"
run_model "vgg16" "3"