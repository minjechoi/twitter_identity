# taco
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity gender_women
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity gender_women
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity gender_nonbinary
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity gender_nonbinary
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity occupation_services
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity occupation_services
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity occupation_tech
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity occupation_tech
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity age_13-17
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity age_13-17
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity age_18-24
# CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity age_18-24
CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity religion_cathchrist --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity religion_cathchrist --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type retweet --identity religion_atheism --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=0 python train.py --tweet_type tweet --identity religion_atheism --train_batch_size 32 --eval_batch_size 32


# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity ethnicity_latin --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity ethnicity_latin --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity gender_men --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity gender_men --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity occupation_writing --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity occupation_writing --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity personal_sensitive --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity personal_sensitive --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity relationship_sibling --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity relationship_sibling --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity personal_socialmedia --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity personal_socialmedia --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity political_blm --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity political_blm --train_batch_size 32 --eval_batch_size 32
# CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity relationship_parent
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity religion_general --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity religion_general --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity religion_islam --train_batch_size 32 --eval_batch_size 32
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity religion_islam --train_batch_size 32 --eval_batch_size 32

CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type retweet --identity religion_hinduism
CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type tweet --identity religion_hinduism

# burger
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity ethnicity_asian
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity ethnicity_asian
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity ethnicity_hispanic
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity ethnicity_hispanic
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity occupation_influencer
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity occupation_influencer
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity occupation_news
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity occupation_news
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity political_conservative
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity political_conservative
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity political_liberal
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity political_liberal

# bagel
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity age_25-34
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity age_25-34
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type retweet --identity age_35-49
CUDA_VISIBLE_DEVICES=1 python train.py --tweet_type tweet --identity age_35-49

CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type retweet --identity age_50+
CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type tweet --identity age_50+
CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type retweet --identity ethnicity_african
CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type tweet --identity ethnicity_african
CUDA_VISIBLE_DEVICES=2 python train.py --tweet_type tweet --identity relationship_parent

CUDA_VISIBLE_DEVICES=3 python train.py --tweet_type retweet --identity occupation_academia
CUDA_VISIBLE_DEVICES=3 python train.py --tweet_type tweet --identity occupation_academia
CUDA_VISIBLE_DEVICES=3 python train.py --tweet_type retweet --identity occupation_art
CUDA_VISIBLE_DEVICES=3 python train.py --tweet_type tweet --identity occupation_art
CUDA_VISIBLE_DEVICES=3 python train.py --tweet_type retweet --identity relationship_partner

CUDA_VISIBLE_DEVICES=4 python train.py --tweet_type retweet --identity occupation_business
CUDA_VISIBLE_DEVICES=4 python train.py --tweet_type tweet --identity occupation_business
CUDA_VISIBLE_DEVICES=4 python train.py --tweet_type retweet --identity occupation_healthcare
CUDA_VISIBLE_DEVICES=4 python train.py --tweet_type tweet --identity occupation_healthcare
CUDA_VISIBLE_DEVICES=4 python train.py --tweet_type tweet --identity relationship_partner

