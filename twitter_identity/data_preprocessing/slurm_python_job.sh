#!/bin/bash
# JOB HEADERS HERE

# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=twitter_identity
#SBATCH --mail-user=minje@umich.edu
#SBATCH --account=drom0
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01-02:00:00
#SBATCH --partition=standard
#SBATCH --output=%x-%j.log

module load python3.9-anaconda

echo "***** Spark cluster is running. Access the Web UI at ${SPARK_MASTER_WEBUI}. *****"

# Change executor resources below to match resources requested above
# with an allowance for spark driver overhead.
# Change the path to your spark job.

# python3 collect_tweets_greatlakes.py
python3 extract_identities.py
echo "*** Job completed! ***"