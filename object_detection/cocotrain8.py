import os
import argparse
import subprocess
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch


def train_multi_gpu(config_path, work_dir, resume_from=None, num_gpus=1):
    """多GPU或单GPU训练，自适应"""
    actual_gpu_count = torch.cuda.device_count()
    if actual_gpu_count == 0:
        print("❌ 没有检测到可用的 GPU，请检查环境。")
        return False
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print("🔧 当前 CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', '未设置'))
    if num_gpus > actual_gpu_count:
        print(f"⚠️ 当前请求使用 {num_gpus} 张 GPU，但系统仅检测到 {actual_gpu_count} 张，自动调整为 {actual_gpu_count} 张。")
        num_gpus = actual_gpu_count

    cmd = [
        'torchrun',
        f'--nproc-per-node={num_gpus}',
        'tools/train.py',
        config_path,
        '--work-dir', work_dir
    ]
    if resume_from:
        cmd.extend(['--resume-from', resume_from])

    env = os.environ.copy()
    # env.pop('CUDA_VISIBLE_DEVICES', None)  # 不绑定具体卡
    env['CUDA_VISIBLE_DEVICES'] = '0,1,2'


    env['PYTHONPATH'] = os.getcwd()
    env['NCCL_IB_DISABLE'] = '1'
    env['NCCL_P2P_DISABLE'] = '1'

    cwd = os.getcwd()

    print(f"📌 实际使用 GPU 数量: {num_gpus}")
    print(f"📌 设置 PYTHONPATH = {env['PYTHONPATH']}")
    print(f"📂 切换工作目录: {cwd}")
    print(f"⚙️ 执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, env=env, cwd=cwd, check=True)
        print("✅ 训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return False


def analyze_log_and_plot(work_dir):
    """训练日志分析并生成 loss 与 mAP 曲线"""
    log_path = Path(work_dir) / 'log.json'
    if not log_path.exists():
        print("⚠️ 未找到 log.json，无法生成曲线图")
        return

    epochs = []
    losses = []
    maps = []

    with open(log_path, 'r') as f:
        for line in f:
            log_entry = json.loads(line)
            if log_entry.get('mode') == 'train' and 'loss' in log_entry:
                epochs.append(log_entry['epoch'])
                losses.append(log_entry['loss'])
            elif log_entry.get('mode') == 'val' and 'coco/bbox_mAP' in log_entry:
                maps.append((log_entry['epoch'], log_entry['coco/bbox_mAP']))

    if losses:
        plt.figure()
        plt.plot(epochs[:len(losses)], losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(Path(work_dir) / "loss_curve.jpg")

    if maps:
        map_epochs, map_vals = zip(*maps)
        plt.figure()
        plt.plot(map_epochs, map_vals, marker='o')
        plt.title("Validation mAP@50")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@50")
        plt.grid(True)
        plt.savefig(Path(work_dir) / "mAP_curve.jpg")

        best_map_epoch, best_map = max(maps, key=lambda x: x[1])
        with open(Path(work_dir) / "summary.txt", "w") as f:
            f.write(f"训练完成。\n最优 mAP@50: {best_map:.4f} (Epoch {best_map_epoch})\n")
      
    
    
      
    

    print("✅ 已生成 loss_curve.jpg, mAP_curve.jpg 和 summary.txt")


def main():
    parser = argparse.ArgumentParser(description='自动适配 GPU 训练 + 自动生成 loss/mAP 曲线')
    parser.add_argument('--config',
                        default='configs/mamba_vision/temp.py',
                        help='配置文件路径')
    parser.add_argument('--work-dir',
                        default='./work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco128',
                        help='工作目录')
    parser.add_argument('--resume-from', default=None,
                        help='从指定 checkpoint 恢复训练')
    parser.add_argument('--gpus', type=int, default=None,
                        help='使用的 GPU 数量（默认自动检测）')
    args = parser.parse_args()

    # 自动检测可用 GPU 数
    requested_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 工作目录: {work_dir}")

    start_time = time.time()
    success = train_multi_gpu(args.config, str(work_dir), args.resume_from, num_gpus=requested_gpus)
    if success:
        analyze_log_and_plot(work_dir)
    duration = time.time() - start_time
    print(f"⏱️ 总训练时间: {duration:.2f} 秒")


if __name__ == '__main__':
    main()
