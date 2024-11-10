export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks/gen_qa_pairs.py
export MODEL=gpt-4o
# export CUDA_VISIBLE_DEVICES=4

## Run self-generation for topic index 0
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=0
## Run self-generation for topic index 1
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=1

## Run self-generation for topic index 2
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=2
##
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=3

python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=4
##
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=5

python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=6
##

