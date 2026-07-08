use regex::Regex;
use serde::Serialize;
use std::process::{Command, exit};
use std::{env, thread, time};

#[derive(Debug)]
enum JobPIDs {
    JobFinished,
    PIDs(Vec<u64>),
}

fn get_job_pids(job_id: &str) -> Result<JobPIDs, String> {
    dbg!(&job_id);

    let listpids_output = Command::new("scontrol")
        .arg("listpids")
        .arg(job_id)
        .output()
        .map_err(|err| format!("failed to run listpids command: {err}"))?;

    let stdout = String::from_utf8_lossy(&listpids_output.stdout);
    let stderr = String::from_utf8_lossy(&listpids_output.stderr);

    eprint!("{stdout}");
    // dbg!(&stderr);

    #[allow(clippy::items_after_statements)]
    const FINISHED_JOB_MSG: &str = "There are no steps for job";
    if stderr.contains(FINISHED_JOB_MSG) | stdout.contains(FINISHED_JOB_MSG) {
        // TODO: log at debug level
        eprintln!("Job is finished");
        return Ok(JobPIDs::JobFinished);
    }

    if !listpids_output.status.success() {
        dbg!(listpids_output.status.code());
        exit(listpids_output.status.code().unwrap_or(1));
    }

    if !stderr.is_empty() {
        // TODO: log at warning level
        eprintln!("listpids produced stderr:\n{stderr}");
    }

    // dbg!(&stdout.lines().collect::<Vec<_>>());

    let line_splits = stdout
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();

    // dbg!(&line_splits);

    let pid_idx = line_splits
        .first()
        .ok_or("can't get first line in listpids stdout")?
        .iter()
        .position(|header_entry| *header_entry == "PID")
        .ok_or("can't find PID entry in first line of listpids")?;

    if pid_idx != 0 {
        eprintln!("WARNING: Expected PID column to be 0, but was {pid_idx}");
    }

    let pids: Result<Vec<u64>, String> = line_splits
        .iter()
        .skip(1)
        .map(|line_split| {
            line_split
                .get(pid_idx)
                .ok_or(format!("missing PID column: {line_split:?}"))?
                .parse()
                .map_err(|err| format!("error parsing PID: {err}"))
        })
        .collect();

    let pids = pids?;

    Ok(JobPIDs::PIDs(pids))
}

#[derive(Debug, Serialize)]
struct JobInfo {
    user_name: String,
    user_id: usize,
    job_id: u32,
    job_name: String,
    is_array_task: bool,
    array_job_id: Option<u32>,
    array_task_id: Option<u32>,
    requested_cpus: u16,
    requested_memory: u64,
    is_gpu_job: bool,
    requested_gpu_shards: u8,
    requested_gpus: f32,
    requested_gpu_memory: u64,
}

fn get_first_regex_group(regex: &str, haystack: &str) -> Result<String, String> {
    let regex_obj = Regex::new(regex).map_err(|err| format!("failed to build regex: {err}"))?;

    let captures = regex_obj.captures(haystack).ok_or(format!(
        "failed to match regex '{regex}' in haystack:\n{haystack:?}"
    ))?;

    Ok(captures
        .get(1)
        .ok_or(format!(
            "no groups captured ({captures:?}) for regex '{regex}' in haystack '{haystack:?}'"
        ))?
        .as_str()
        .to_string())
}

#[allow(clippy::similar_names)]
fn get_job_info(job_id: &str) -> Result<JobInfo, String> {
    let scontrol_show_output = Command::new("scontrol")
        .arg("show")
        .arg("jobid")
        .arg("-dd")
        .arg(job_id)
        .output()
        .map_err(|err| format!("failed to run scontrol show command: {err}"))?;

    let stdout = String::from_utf8_lossy(&scontrol_show_output.stdout);
    let stderr = String::from_utf8_lossy(&scontrol_show_output.stderr);

    // dbg!(&stdout);
    // dbg!(&stderr);

    if !stderr.is_empty() {
        // TODO: log at warning level
        eprintln!("scontrol show produced stderr:\n{stderr}");
    }

    // eprint!("{stdout}");

    let is_array_task = stdout.contains("ArrayTaskId");

    let user_name = get_first_regex_group(r"UserId=([a-z_\-]+)\(", &stdout)?;
    let user_id = get_first_regex_group(r"UserId=[a-z_\-]+?\((\d+)\)", &stdout)?
        .parse::<usize>()
        .map_err(|err| format!("failed to parse user_id: {err}"))?;
    let job_id = get_first_regex_group(r"JobId=(\d+)", &stdout)?
        .parse::<u32>()
        .map_err(|err| format!("failed to parse job_id: {err}"))?;
    let job_name = get_first_regex_group(r"JobName=(.+?)\s", &stdout)?;

    let (array_job_id, array_task_id) = if is_array_task {
        (
            Some(
                get_first_regex_group(r"ArrayJobId=(\d+)", &stdout)?
                    .parse::<u32>()
                    .map_err(|err| format!("failed to parse array_job_id: {err}"))?,
            ),
            Some(
                get_first_regex_group(r"ArrayTaskId=(\d+)", &stdout)?
                    .parse::<u32>()
                    .map_err(|err| format!("failed to parse array_task_id: {err}"))?,
            ),
        )
    } else {
        (None, None)
    };

    let requested_cpus = get_first_regex_group(r"\sNumCPUs=(\d+)", &stdout)?
        .parse::<u16>()
        .map_err(|err| format!("failed to parse requested_cpus: {err}"))?;

    // this is always in megabytes I think
    let requested_memory = get_first_regex_group(r"\sMem=(\d+)", &stdout)?
        .parse::<u64>()
        .map_err(|err| format!("failed to parse requested_memory: {err}"))?;

    let is_gpu_job = stdout
        .lines()
        .find(|line| line.contains("JOB_GRES"))
        .ok_or(format!("could not find JOB_GRES line in '{stdout}'"))?
        .contains("shard");

    let (requested_gpu_shards, requested_gpus, requested_gpu_memory) = if is_gpu_job {
        let requested_gpu_shards = get_first_regex_group(r"\sJOB_GRES=.*?shards:(\d+)", &stdout)?
            .parse::<u8>()
            .map_err(|err| format!("failed to parse requested_gpu_shards: {err}"))?;

        (requested_gpu_shards, 0.0, 0)
    } else {
        (0, 0.0, 0)
    };

    // let requested_gpu_shards = get_first_regex_group(r"\sJOB_GRES", haystack)

    Ok(JobInfo {
        user_name,
        user_id,
        job_id,
        job_name,
        is_array_task,
        array_job_id,
        array_task_id,
        requested_cpus,
        requested_memory,
        is_gpu_job,
        requested_gpu_shards,
        requested_gpus,
        requested_gpu_memory,
    })
}

fn main() -> Result<(), String> {
    let Ok(job_id) = env::var("SLURM_JOB_ID") else {
        return Err("SLURM_JOB_ID is not set".to_string());
    };
    dbg!(&job_id);

    let job_info = get_job_info(&job_id)?;

    dbg!(&job_info);

    while let JobPIDs::PIDs(pids) = get_job_pids(&job_id)? {
        dbg!(pids);
        thread::sleep(time::Duration::from_millis(500));
    }

    Ok(())
}
