import os
import argparse
import subprocess
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch


def train_multi_gpu(config_path, work_dir, resume_from=None, num_gpus=1):
    """Multi-GPU or Single-GPU training with auto-adaptation"""
    actual_gpu_count = torch.cuda.device_count()
    
    if actual_gpu_count == 0:
        print("❌ No available GPUs detected. Please check your environment.")
        return False

    for i in range(actual_gpu_count):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("🔧 Current CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))

    if num_gpus > actual_gpu_count:
        print(f"⚠️ Requested {num_gpus} GPUs, but only {actual_gpu_count} detected. Adjusting to {actual_gpu_count}.")
        num_gpus = actual_gpu_count

    # Construct the torchrun command
    cmd = [
        'torchrun',
        f'--nproc-per-node={num_gpus}',
        'tools/train.py',
        config_path,
        '--work-dir', work_dir
    ]
    
    if resume_from:
        cmd.extend(['--resume-from', resume_from])

    # Setup Environment Variables
    env = os.environ.copy()
    # Optional: Hardcode specific GPUs if necessary
    # env['CUDA_VISIBLE_DEVICES'] = '0,1,2' 

    env['PYTHONPATH'] = os.getcwd()
    env['NCCL_IB_DISABLE'] = '1'
    env['NCCL_P2P_DISABLE'] = '1'

    cwd = os.getcwd()

    print(f"📌 Actual GPUs in use: {num_gpus}")
    print(f"📌 PYTHONPATH set to: {env['PYTHONPATH']}")
    print(f"📂 Working Directory: {cwd}")
    print(f"⚙️ Executing Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env, cwd=cwd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False


def analyze_log_and_plot(work_dir):
    """Analyze training logs and generate Loss and mAP curves"""
    log_path = Path(work_dir) / 'log.json'
    if not log_path.exists():
        print("⚠️ log.json not found. Cannot generate curves.")
        return

    epochs = []
    losses = []
    maps = []

    try:
        with open(log_path, 'r') as f:
            for line in f:
                log_entry = json.loads(line)
                if log_entry.get('mode') == 'train' and 'loss' in log_entry:
                    epochs.append(log_entry['epoch'])
                    losses.append(log_entry['loss'])
                elif log_entry.get('mode') == 'val' and 'coco/bbox_mAP' in log_entry:
                    maps.append((log_entry['epoch'], log_entry['coco/bbox_mAP']))
    except Exception as e:
        print(f"❌ Error parsing log file: {e}")
        return

    # Plotting Training Loss
    if losses:
        plt.figure()
        plt.plot(epochs[:len(losses)], losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(Path(work_dir) / "loss_curve.jpg")
        plt.close()

    # Plotting Validation mAP
    if maps:
        map_epochs, map_vals = zip(*maps)
        plt.figure()
        plt.plot(map_epochs, map_vals, marker='o')
        plt.title("Validation mAP@50")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@50")
        plt.grid(True)
        plt.savefig(Path(work_dir) / "mAP_curve.jpg")
        plt.close()

        # Save Summary
        best_map_epoch, best_map = max(maps, key=lambda x: x[1])
        summary_path = Path(work_dir) / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("Training Summary\n")
            f.write(f"Best mAP@50: {best_map:.4f} (at Epoch {best_map_epoch})\n")
    
    print("✅ Generated loss_curve.jpg, mAP_curve.jpg, and summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Auto-adaptive GPU Training + Loss/mAP Visualization')
    parser.add_argument('--config',
                        default='configs/mamba_vision/temp.py',
                        help='Path to the configuration file')
    parser.add_argument('--work-dir',
                        default='./work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco128',
                        help='Directory to save logs and models')
    parser.add_argument('--resume-from', default=None,
                        help='Resume training from a specific checkpoint')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect)')
    args = parser.parse_args()

    # Auto-detect available GPUs
    requested_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Target Work Directory: {work_dir}")

    start_time = time.time()
    
    # Execute training
    success = train_multi_gpu(
        args.config, 
        str(work_dir), 
        args.resume_from, 
        num_gpus=requested_gpus
    )
    
    # Post-training analysis
    if success:
        analyze_log_and_plot(work_dir)
    
    duration = time.time() - start_time
    print(f"⏱️ Total training duration: {duration:.2f} seconds")


if __name__ == '__main__':
    main()
