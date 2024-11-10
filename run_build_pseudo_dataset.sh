export PYTHONPATH=$PYTHONPATH:../

python tasks/build_pseudo_dataset.py \
    --data_dir=data/self-prompt-cot/gpt-3.5-turbo/ \
    --output_dir=data/self-prompt-cot \
    --demo_path=demos/multihop_demos.json \
    --yesno_demo_path=demos/yesno_demos.json \
    --do_completion \
    --do_merge
