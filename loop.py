import submitit

def loop():
    a = 1
    while True:
        a += 1

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder='/Utilisateurs/wsun01/logs/')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        job_name='ocr',
        timeout_min=2160 * 4,
        gpus_per_node=1,
        cpus_per_task=5,
        mem_gb=40 * 2,
        # slurm_partition='gpu-a6000',
        slurm_additional_parameters={
            'nodelist': 'l3icalcul10'
        }
    )
    executor.submit(loop)