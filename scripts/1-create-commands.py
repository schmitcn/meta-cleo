import random

TARGET_DIRECTORY = "ensemble-learners"
SCRIPT_PATH = "scripts/2-generate-ensemble-from-openml-task.py"
COMMAND_TEMPLATE = "python3 {} -t {} -e {} -m {} -f {} -s {} -d {}"

if __name__ == '__main__':

    # configurations
    # last tasks in descending order of task_id
    tasks_ids = [75199, 75187, 75185, 75180, 75176, 75171, 75159, 75156, 75154,
                 75149, 75148, 75147,
                 75146, 75142]
    ensemble_sizes = [5, 10, 15, 20]
    budget_m = 2000  # 2000 trials
    budget_f = 100  # 100 trials
    runs_per_config = 3
    seed_value = 10
    random.seed(seed_value)

    with open("commands.txt", "w") as f:
        for task_id in tasks_ids:
            for ensemble_size in ensemble_sizes:
                for i in range(0, runs_per_config):
                    f.write(COMMAND_TEMPLATE.format(
                        SCRIPT_PATH,
                        task_id,
                        ensemble_size,
                        budget_m,
                        budget_f,
                        random.randint(0, 1000),
                        TARGET_DIRECTORY))
                    f.write('\n')
