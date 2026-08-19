use chrono::NaiveDateTime;
use clap::Parser;
use gethostname::gethostname;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::struct_wrappers::device::{ProcessInfo, ProcessUtilizationSample};
use nvml_wrapper::{Device, Nvml};
use regex::Regex;
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::fs::{create_dir_all, exists, read_to_string};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{dbg, env, thread};

const SHARDS_PER_GPU: u16 = 2;

#[derive(Parser, Debug)]
struct Args {
    #[arg(
        short = 'o',
        long = "output-dir",
        required = true,
        help = "Directory to write monitoring output"
    )]
    output_dir: String,

    #[arg(
        short = 'i',
        long = "interval-seconds",
        required = false,
        default_value = "0.5",
        help = "Monitoring interval in seconds (must be between 0 and 300)"
    )]
    interval: f32,
}

#[derive(Debug)]
enum JobPIDs {
    JobFinished,
    PIDs(Vec<u32>),
}

fn get_job_pids(job_id: u32) -> Result<JobPIDs, String> {
    let listpids_output = Command::new("scontrol")
        .arg("listpids")
        .arg(job_id.to_string())
        .output()
        .map_err(|err| format!("failed to run listpids command: {err}"))?;

    let stdout = String::from_utf8_lossy(&listpids_output.stdout);
    let stderr = String::from_utf8_lossy(&listpids_output.stderr);

    // eprint!("{stdout}");
    // eprint!("{stderr}");

    #[allow(clippy::items_after_statements)]
    const FINISHED_JOB_MSG: &str = "There are no steps for job";
    if stderr.contains(FINISHED_JOB_MSG) | stdout.contains(FINISHED_JOB_MSG) {
        // TODO: log at debug level
        eprintln!("Job is finished");
        return Ok(JobPIDs::JobFinished);
    }

    if !listpids_output.status.success() {
        let return_code = listpids_output.status.code();
        return Err(format!("listpids exited with code {return_code:?}"));
    }

    if !stderr.is_empty() {
        // TODO: log at warning level
        eprintln!("listpids produced stderr:\n{stderr}");
    }

    let line_splits = stdout
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let pid_idx = line_splits
        .first()
        .ok_or("can't get first line in listpids stdout")?
        .iter()
        .position(|header_entry| *header_entry == "PID")
        .ok_or("can't find PID entry in first line of listpids")?;

    if pid_idx != 0 {
        eprintln!("WARNING: Expected PID column to be 0, but was {pid_idx}");
    }

    let pids: Result<Vec<u32>, String> = line_splits
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
    job_id: u32,
    job_name: String,
    user_name: String,
    user_id: usize,
    submit_time: i64,
    start_time: i64,
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
        "failed to match regex '{regex}' in haystack: '{haystack}'"
    ))?;

    let captured_string = captures
        .get(1)
        .ok_or(format!(
            "no groups captured ({captures:?}) for regex '{regex}' in haystack '{haystack}'"
        ))?
        .as_str()
        .to_string();

    Ok(captured_string)
}

fn get_total_gpu_memory(gpu_index: u32) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|err| format!("unable to initialize NVML: {err}"))?;

    let device = nvml
        .device_by_index(gpu_index)
        .map_err(|err| format!("failed to get NVML device at index {gpu_index}: {err}"))?;

    // this is always in bytes I think
    let total_memory = device
        .memory_info()
        .map_err(|err| format!("failed to get GPU memory information: {err}"))?
        .total;

    Ok(total_memory)
}

#[allow(clippy::similar_names, clippy::items_after_statements)]
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

    // eprint!("{stdout}");
    // eprint!("{stderr}");

    if !stderr.is_empty() {
        // TODO: log at warning level
        eprintln!("scontrol show produced stderr:\n{stderr}");
    }

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

    const TIME_REGEX: &str = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}";
    const TIME_FORMAT: &str = "%Y-%m-%dT%H:%M:%S";

    let submit_time = get_first_regex_group(&format!("SubmitTime=({TIME_REGEX})"), &stdout)?;
    let submit_time = NaiveDateTime::parse_from_str(&submit_time, TIME_FORMAT)
        .map_err(|err| format!("unable to parse date '{submit_time}': {err}"))?
        .and_utc()
        .timestamp();

    let start_time = get_first_regex_group(&format!("StartTime=({TIME_REGEX})"), &stdout)?;
    let start_time = NaiveDateTime::parse_from_str(&start_time, TIME_FORMAT)
        .map_err(|err| format!("unable to parse date '{start_time}': {err}"))?
        .and_utc()
        .timestamp();

    let requested_cpus = get_first_regex_group(r"NumCPUs=(\d+)", &stdout)?
        .parse::<u16>()
        .map_err(|err| format!("failed to parse requested_cpus: {err}"))?;

    // this is always in megabytes I think
    let mut requested_memory = get_first_regex_group(r"Mem=(\d+)", &stdout)?
        .parse::<u64>()
        .map_err(|err| format!("failed to parse requested_memory: {err}"))?;

    // to bytes
    requested_memory *= 10_u64.pow(6);

    let is_gpu_job = stdout
        .lines()
        .find(|line| line.contains("JOB_GRES"))
        .ok_or(format!("could not find JOB_GRES line in '{stdout}'"))?
        .contains("shard");

    let (requested_gpu_shards, requested_gpus, requested_gpu_memory) = if is_gpu_job {
        let requested_gpu_shards = get_first_regex_group(r"JOB_GRES=.*?shard:(\d+)", &stdout)?
            .parse::<u8>()
            .map_err(|err| format!("failed to parse requested_gpu_shards: {err}"))?;

        // because of sharding we can have e.g. half of a gpu reserved
        let requested_gpus = f32::from(requested_gpu_shards) / f32::from(SHARDS_PER_GPU);

        let total_gpu_memory = get_total_gpu_memory(0)?; // assume all GPUs are equal

        #[allow(clippy::cast_precision_loss)]
        let requested_gpu_memory = total_gpu_memory as f64 * f64::from(requested_gpus);

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let requested_gpu_memory = requested_gpu_memory as u64;

        (requested_gpu_shards, requested_gpus, requested_gpu_memory)
    } else {
        (0, 0.0, 0)
    };

    Ok(JobInfo {
        job_id,
        job_name,
        user_name,
        user_id,
        submit_time,
        start_time,
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

fn unix_timestamp() -> f64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("you are a time traveller?");

    #[allow(clippy::cast_precision_loss)]
    let timestamp = duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;

    timestamp
}

#[derive(Debug)]
struct JobResourceUsage {
    // number of microseconds of cpu usage since the beginning of the job
    cpu_utilization: u64,

    // number of bytes used by the process at the point in time it was measured
    memory: u64,

    // GPU utilization as reported by NVML (same as nvidia-smi). It's defined a bit vagely in the documentation
    gpu_utilization: u32,

    // number of bytes of GPU memory used as reported by NVML (sum of individual processes)
    gpu_memory: u64,
}

fn gpu_utilization_stats(pids: &[u32]) -> Result<(u32, u64), String> {
    let nvml = Nvml::init().map_err(|err| format!("unable to initialize NVML: {err}"))?;

    let n_gpus = nvml
        .device_count()
        .map_err(|err| format!("unable to get GPU count: {err}"))?;

    let gpu_devices = (0..n_gpus)
        .map(|gpu_idx| {
            nvml.device_by_index(gpu_idx)
                .map_err(|err| format!("unable to get GPU device at index {gpu_idx}: {err}"))
        })
        .collect::<Result<Vec<Device<'_>>, String>>()?;

    let pids_hash: HashSet<u32> = pids.iter().copied().collect();

    let total_gpu_memory_usage: u64 = gpu_devices
        .iter()
        .map(|device| {
            device.running_compute_processes().map_err(|err| {
                format!("unable to get compute processes for device {device:?}: {err}")
            })
        })
        .collect::<Result<Vec<Vec<ProcessInfo>>, String>>()?
        .into_iter()
        .flatten()
        .filter(|proc_info| pids_hash.contains(&proc_info.pid))
        .map(|proc_info| match proc_info.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => Ok(bytes),
            UsedGpuMemory::Unavailable => Err(format!(
                "GPU memory usage unavailable for process {}",
                &proc_info.pid
            )),
        })
        .collect::<Result<Vec<u64>, String>>()?
        .iter()
        .sum();

    dbg!(total_gpu_memory_usage);

    let total_gpu_utilization: u32 = gpu_devices
        .iter()
        .map(|device| {
            device.process_utilization_stats(None).map_err(|err| {
                format!("unable to get process utilization stats for device {device:?}: {err}")
            })
        })
        .collect::<Result<Vec<Vec<ProcessUtilizationSample>>, String>>()?
        .into_iter()
        .flatten()
        .filter(|proc_util_stats| pids_hash.contains(&proc_util_stats.pid))
        .map(|proc_util_stats| proc_util_stats.sm_util)
        .sum();

    Ok((total_gpu_utilization, total_gpu_memory_usage))
}

fn get_cgroup_directory(job_id: u32) -> Result<String, String> {
    let node = gethostname();
    let node = node.to_str().ok_or("failed to get machine hostname")?;

    let path = format!("/sys/fs/cgroup/system.slice/{node}_slurmstepd.scope/job_{job_id}");

    let path_exists = exists(&path).map_err(|err| {
        format!("unable to check existence cgroup directory for job {job_id} '{path}': {err}")
    })?;

    if !path_exists {
        return Err(format!(
            "cgroup directory for job {job_id} '{path}' does not exist"
        ));
    }

    Ok(path)
}

fn get_memory_usage(job_id: u32) -> Result<u64, String> {
    // get_cgroup_directory asserts existence so we can just assume that memory.current and memory.stat will exist in
    // there
    let cgroup_dir = get_cgroup_directory(job_id)?;

    let memory_stat_path = format!("{cgroup_dir}/memory.stat");
    let current_memory_path = format!("{cgroup_dir}/memory.current");

    let mem_stat_contents = read_to_string(&memory_stat_path)
        .map_err(|err| format!("failed to read {memory_stat_path}: {err}"))?;

    let mut current_mem_contents = read_to_string(&current_memory_path)
        .map_err(|err| format!("failed to read {current_memory_path}: {err}"))?;

    current_mem_contents.retain(|c| !c.is_whitespace());

    let current_mem: u64 = current_mem_contents
        .parse()
        .map_err(|err| {
            format!("unable to parse current memory cgroup file contents '{current_mem_contents}' into u64: {err}")
        })?;

    let inactive_file_mem: u64 = mem_stat_contents
        .lines()
        .find(|line| line.starts_with("inactive_file"))
        .ok_or("could not find inactive_file line")?
        .split_whitespace()
        .nth(1)
        .ok_or("unable to get inactive_file value")?
        .parse()
        .map_err(|err| format!("unable to parse inactive_file value: {err}"))?;

    if inactive_file_mem > current_mem {
        return Err(format!(
            "Inactive file memory ({inactive_file_mem}) is larger than  current memory({current_mem})"
        ));
    }

    // calculate Working Set Size as memory.current - memory.stat:inactive_file, same as cAdvisor does (see explanation:
    // https://itnext.io/from-rss-to-wss-navigating-the-depths-of-kubernetes-memory-metrics-4d7d77d8fdcb#0555 )
    let mem_usage = current_mem - inactive_file_mem;

    Ok(mem_usage)
}

fn get_cpu_usage(job_id: u32) -> Result<u64, String> {
    let cgroup_dir = get_cgroup_directory(job_id)?;

    // get_cgroup_directory asserts existence so we can just assume that cpu.stat will exist in there
    let cpu_stat_path = format!("{cgroup_dir}/cpu.stat");

    let usage_usec: u64 = read_to_string(&cpu_stat_path)
        .map_err(|err| format!("unable to read cpu stat file '{cpu_stat_path}': {err}"))?
        .lines()
        .find(|line| line.starts_with("usage_usec "))
        .ok_or("could not find usage_usec line")?
        .split_whitespace()
        .nth(1)
        .ok_or("unable to get usage_usec value")?
        .parse()
        .map_err(|err| format!("unable to parse usage_usec: {err}"))?;

    Ok(usage_usec)
}

fn get_job_resource_usage(job_info: &JobInfo, pids: &[u32]) -> Result<JobResourceUsage, String> {
    let cpu_utilization = get_cpu_usage(job_info.job_id)?;
    let memory = get_memory_usage(job_info.job_id)?;

    let (gpu_utilization, gpu_memory) = if job_info.is_gpu_job {
        gpu_utilization_stats(pids)?
    } else {
        (0, 0)
    };

    Ok(JobResourceUsage {
        cpu_utilization,
        memory,
        gpu_utilization,
        gpu_memory,
    })
}

fn log_usage_stats(
    timestamp: f64,
    job_resource_usage: &JobResourceUsage,
    csv_writer: &mut csv::Writer<fs::File>,
) -> Result<(), String> {
    csv_writer
        .write_record(vec![
            timestamp.to_string(),
            job_resource_usage.cpu_utilization.to_string(),
            job_resource_usage.memory.to_string(),
            job_resource_usage.gpu_utilization.to_string(),
            job_resource_usage.gpu_memory.to_string(),
        ])
        .map_err(|err| format!("failed to write CSV record: {err}"))?;

    csv_writer
        .flush()
        .map_err(|err| format!("failed to flush CSV writer buffer: {err}"))?;

    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    if args.interval < 0.0 || args.interval > 300.0 {
        return Err(format!(
            "monitoring interval ({}) must be between 0 and 300 seconds",
            args.interval
        ));
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let sleep_interval = Duration::from_millis((args.interval * 1000.0) as u64);

    let Ok(job_id) = env::var("SLURM_JOB_ID") else {
        return Err("SLURM_JOB_ID is not set".to_string());
    };
    dbg!(&job_id);

    let job_info = get_job_info(&job_id)?;

    dbg!(&job_info);

    print!("{}", serde_json::to_string_pretty(&job_info).unwrap());

    create_dir_all(&args.output_dir).map_err(|err| {
        format!(
            "failed to create output directory '{}': {}",
            args.output_dir, err
        )
    })?;

    let mut csv_writer = csv::Writer::from_path(format!("{}/usage_stats.csv", args.output_dir))
        .map_err(|err| format!("unable to create CSV file: {err}"))?;

    csv_writer
        .write_record(vec![
            "timestamp",
            "cpu_usage",
            "memory",
            "gpu_usage",
            "gpu_memory",
        ])
        .map_err(|err| format!("failed to write output CSV header: {err}"))?;

    while let JobPIDs::PIDs(pids) = get_job_pids(job_info.job_id)? {
        dbg!(&pids);
        let timestamp = unix_timestamp();

        dbg!(timestamp);
        let job_resource_usage = get_job_resource_usage(&job_info, &pids)?;
        dbg!(&job_resource_usage);

        log_usage_stats(timestamp, &job_resource_usage, &mut csv_writer)?;

        thread::sleep(sleep_interval);
    }

    Ok(())
}
