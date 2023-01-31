python train.py \
    --train_file /shared/3/projects/bio-change/data/processed/identity_classifier-train_data/retweets_quotes.age_13-17.train.tsv.gz \
    --test_file /shared/3/projects/bio-change/data/processed/identity_classifier-train_data/retweets_quotes.age_13-17.test.tsv.gz \
    --val_file /shared/3/projects/bio-change/data/processed/identity_classifier-train_data/retweets_quotes.age_13-17.val.tsv.gz \
    --default_root_dir /shared/3/projects/bio-change/results/experiments/identity-classifier/retweets_quotes/age_13-17 \
    --model_name_or_path cardiffnlp/tweet-topic-21-multi \
    --model_cache_dir /shared/3/projects/bio-change/.cache/ \
    --precision 16 \
    --train_batch_size 8 \
    --max_epochs 10 \
    --weighted_class_loss \
    --num_workers 4 \
    --patience 1 \
    --accelerator gpu \
    --devices 1

