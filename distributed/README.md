# Distributed Computing Resources
Resources for running on distributed computing systems, such as Hipergator.

`train_hpg.sh` is a sample script to run a job for a Python file on Hipergator. Please update the paths where indicated to your paths. You can submit the job to be run on the resources with `sbatch train_hpg.sh`. To check the status of jobs running in our groups, use `squeue -A prabhat` or `squeuemine`. To cancel a job, use `scancel <jobid>` where jobid is the job ID given to your job on submittal. You can find the ID in squeue as well.