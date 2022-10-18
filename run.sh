pip install transformers==4.23.1

python inference.py --num_iter 100 --num_warmup 5 --channels_last 1 --device cuda --precision float16 --jit --nv_fuser --batch_size 16
