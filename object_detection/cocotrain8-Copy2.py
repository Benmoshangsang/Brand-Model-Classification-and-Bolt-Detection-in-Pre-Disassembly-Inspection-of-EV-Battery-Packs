# -*- coding: utf-8 -*-
import os
import argparse
import subprocess
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch


def train_multi_gpu(config_path, work_dir, resume_from=None, num_gpus=1, batch_size=None, accum_steps=None, extra_args=None):
    """
    使用 torchrun 进行多GPU或单GPU训练，并支持动态调整 batch_size 和梯度累积。
    ✅【功能增强】支持传递额外的未知参数给底层训练脚本。

    Args:
        config_path (str): MMDetection 配置文件的路径。
        work_dir (str): 实验的工作目录，用于保存日志和模型。
        resume_from (str, optional): 从指定的 checkpoint 文件恢复训练。默认为 None。
        num_gpus (int): 希望使用的 GPU 数量。
        batch_size (int, optional): 在命令行覆盖配置文件中的每个 GPU 的 batch_size。默认为 None。
        accum_steps (int, optional): 梯度累积的步数。默认为 None (不使用)。
        extra_args (list, optional): 额外的、未被此脚本解析的参数列表，将直接传递给 tools/train.py。

    Returns:
        bool: 训练是否成功。
    """
    # --- GPU 环境检测 ---
    actual_gpu_count = torch.cuda.device_count()  # 获取系统中实际可用的 GPU 总数
    if actual_gpu_count == 0:  # 如果没有任何可用的 GPU
        print("❌ 没有检测到可用的 GPU，请检查 PyTorch 和 CUDA 环境。")  # 打印错误信息
        return False  # 提前返回，表示失败

    print("✅ 检测到以下可用 GPU:")  # 打印提示信息
    for i in range(actual_gpu_count):  # 遍历所有可用的 GPU
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")  # 打印每个 GPU 的名称

    # 检查外部是否设置了 CUDA_VISIBLE_DEVICES
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')  # 获取环境变量
    if visible_devices:  # 如果用户在外部设置了该环境变量
        print(f"🔧 检测到外部环境变量 CUDA_VISIBLE_DEVICES = '{visible_devices}'")  # 打印检测到的设置
        # 根据用户设置的可见设备列表，重新计算可用的 GPU 数量
        # 注意：这里的计算仅用于信息展示和num_gpus的校验，torchrun会自行处理具体的设备映射
        visible_gpu_indices = [g.strip() for g in visible_devices.split(',')]
        actual_gpu_count = len(visible_gpu_indices)
        print(f"    因此，脚本实际可用的 GPU 数量为 {actual_gpu_count} 张。") # 打印调整后的可用数量
    else:  # 如果用户未设置
        print("🔧 未设置 CUDA_VISIBLE_DEVICES，脚本将默认使用所有检测到的物理 GPU。")  # 打印提示

    # --- GPU 数量自适应调整 ---
    if num_gpus > actual_gpu_count:  # 如果请求的 GPU 数量超过了实际可用的数量
        print(f"⚠️  警告: 您请求使用 {num_gpus} 张 GPU，但系统当前可用 {actual_gpu_count} 张。")  # 打印警告
        print(f"    将自动调整为使用 {actual_gpu_count} 张 GPU。")  # 打印调整信息
        num_gpus = actual_gpu_count  # 将请求数量调整为实际可用数量

    # --- 构建 torchrun 训练命令 ---
    cmd = [
        'torchrun',  # 使用 torchrun 作为分布式启动器
        f'--nproc-per-node={num_gpus}',  # 设置每个节点的进程数，即使用的 GPU 数量
        'tools/train.py',  # MMDetection 的训练脚本
        config_path,  # 传入配置文件路径
        '--launcher', 'pytorch',  # ✅ 【关键修复调整】将 launcher 参数紧跟在 train.py 之后，确保被正确解析
        '--work-dir', work_dir,  # 传入工作目录
    ]

    # --- 可选参数处理 ---
    if resume_from:  # 如果指定了恢复训练的 checkpoint
        cmd.extend(['--resume-from', resume_from])  # 将恢复训练的参数添加到命令中
    
    # 用于收集所有需要动态修改的配置项
    cfg_options = []

    # ✅ 功能增强：动态修改 batch_size 以避免 OOM
    if batch_size is not None:  # 如果用户通过命令行传入了 batch_size
        print(f"🔧 将通过命令行覆盖配置文件中的 batch_size，设置为每张 GPU {batch_size}。") # 打印提示
        # 注意: 'train_dataloader.batch_size' 是 MMDetection 中常见的配置路径
        cfg_options.append(f'train_dataloader.batch_size={batch_size}')

    # ✅ 【新增大杀器】梯度累积 (Gradient Accumulation)
    if accum_steps is not None and accum_steps > 1:
        print(f"🚀 启用梯度累积！累积步数为 {accum_steps}。") # 打印提示
        # MMEngine 通过 optim_wrapper.accumulative_counts 来实现
        cfg_options.append(f'optim_wrapper.accumulative_counts={accum_steps}')
        # 计算等效的全局 batch_size
        effective_batch_size_per_gpu = (batch_size if batch_size is not None else "config_bs")
        print(f"    等效全局 Batch Size ≈ {num_gpus} GPUs * {effective_batch_size_per_gpu} (per GPU) * {accum_steps} accum = {num_gpus * (batch_size if batch_size is not None else -1)} (approx)")

    if cfg_options: # 如果有任何需要修改的配置
        cmd.extend(['--cfg-options'] + cfg_options)
    
    # ✅【核心修复】将所有未被解析的额外参数追加到命令末尾
    if extra_args:
        print(f"🔧 检测到并传递额外参数: {' '.join(extra_args)}")
        cmd.extend(extra_args)

    # --- 设置子进程环境变量 ---
    env = os.environ.copy()  # 复制当前的环境变量，以便进行修改
    env['PYTHONPATH'] = os.getcwd()  # 将当前目录添加到 PYTHONPATH，确保能找到项目内的模块
    env['NCCL_P2P_DISABLE'] = '1'    # 禁用 GPU 间的 P2P 直接通信，在某些云环境或docker中可以提高稳定性
    env['NCCL_IB_DISABLE'] = '1'     # 在某些网络环境下，禁用 InfiniBand 可以避免 NCCL 初始化问题

    cwd = os.getcwd()  # 获取当前工作目录

    # --- 打印最终执行信息 ---
    print("-" * 50) # 打印分隔符，使输出更清晰
    print(f"📌 最终使用的 GPU 数量: {num_gpus}")  # 打印实际使用的 GPU 数量
    print(f"📌 继承的 CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '未设置')}") # 打印继承的环境变量
    print(f"📌 设置 PYTHONPATH = {env['PYTHONPATH']}")  # 打印设置的 PYTHONPATH
    print(f"📂 脚本将在以下目录执行: {cwd}")  # 打印执行目录
    print(f"⚙️  将执行以下命令:")  # 打印命令标题
    # 使用 repr() 来显示命令列表，更清晰地区分每个参数
    print(f"    {' '.join(cmd)}")
    print("-" * 50) # 打印分隔符

    # --- 执行训练命令 ---
    try:
        # 使用 subprocess.run 执行命令，并设置 check=True，如果命令失败会抛出异常
        subprocess.run(cmd, env=env, cwd=cwd, check=True, text=True) # text=True 使得输出更友好
        print("\n✅ 训练过程成功完成!")  # 训练成功
        return True  # 返回成功状态
    except subprocess.CalledProcessError as e:  # 如果命令执行失败 (例如 OOM)
        print(f"\n❌ 训练过程失败: {e}")  # 打印错误信息
        print("    请检查上面的日志输出以定位具体问题。常见的可能原因是：") # 给出调试建议
        print("    1. 显存不足 (CUDA out of memory)。尝试降低 --batch-size (例如 --bs 1)，并/或增加 --accum-steps (例如 --accum 4)。")
        print("    2. 配置文件路径或内容错误。")
        print("    3. 数据集路径或格式问题。")
        return False  # 返回失败状态
    except FileNotFoundError: # 如果找不到 torchrun 或 tools/train.py
        print(f"\n❌ 命令执行失败：找不到 `torchrun` 或 `tools/train.py`。") # 打印文件找不到的错误
        print("    请确保您已正确安装 PyTorch 并且当前位于项目根目录下。")
        return False


def analyze_log_and_plot(work_dir):
    """
    ✅ 【修复版】训练日志分析并生成 loss 与 mAP 曲线。
    此函数现在会解析 MMEngine 默认生成的 `scalars.json` 文件。
    """
    # --- 路径查找 ---
    # 寻找最新的时间戳目录
    work_dir_path = Path(work_dir)                       # 将工作目录转换为 Path 对象
    latest_log_dir = None                                # 初始化最新日志目录为 None
    
    # MMEngine 会将日志文件放在 work_dir/timestamp/ 目录下
    # 我们需要找到最新的一个 timestamp 目录
    dirs = [d for d in work_dir_path.iterdir() if d.is_dir()] # 获取工作目录下的所有子目录
    if dirs:                                             # 如果存在子目录
        latest_log_dir = max(dirs, key=os.path.getmtime)   # 根据修改时间找到最新的目录
    
    if not latest_log_dir:                               # 如果没有找到任何日志目录
        print(f"⚠️ 在 {work_dir} 中未找到任何日志目录。")
        return

    # ✅ 日志文件路径更新为 MMEngine 的标准 `scalars.json`
    log_path = latest_log_dir / 'vis_data' / 'scalars.json' # 构建正确的日志文件路径

    if not log_path.exists():                            # 检查日志文件是否存在
        print(f"⚠️ 在 {log_path.parent} 中未找到 {log_path.name}，无法生成曲线图。")
        return

    # --- 数据提取 ---
    steps_loss = []                                      # 用于存储训练 loss 的步骤
    losses = []                                          # 用于存储训练 loss 的值
    epochs_map = []                                      # 用于存储验证 mAP 的 epoch
    maps_val = []                                        # 用于存储验证 mAP 的值

    print(f"🔎 正在解析日志文件: {log_path}")
    try:
        with open(log_path, 'r') as f:                   # 打开日志文件
            for line in f:                               #逐行读取
                log_entry = json.loads(line)             # 将每一行解析为 JSON 对象
                
                # 提取训练 loss
                # MMEngine 的 scalars.json 中，训练时的 loss 通常以 'loss' 为键
                if 'loss' in log_entry:
                    steps_loss.append(log_entry.get('step'))  # 获取当前步骤
                    losses.append(log_entry['loss'])          # 获取 loss 值
                
                # 提取验证 mAP
                # 验证指标通常带有前缀，如 'coco/bbox_mAP'
                elif 'coco/bbox_mAP' in log_entry:
                    # 验证指标通常是按 epoch 记录的，但我们也兼容按 step 记录的情况
                    epochs_map.append(log_entry.get('step')) # MMEngine 中验证指标也记录 step
                    maps_val.append(log_entry['coco/bbox_mAP']) # 获取 mAP 值

        if not losses and not maps_val:                  # 如果没有找到任何 loss 或 mAP 记录
            print("⚠️ 日志文件中没有找到 'loss' 或 'coco/bbox_mAP' 记录。")
            return

        # --- 绘图 ---
        plt.style.use('seaborn-v0_8-whitegrid')          # 设置绘图风格

        # 绘制 Loss 曲线
        if losses:
            plt.figure(figsize=(12, 6))                  # 创建一个新的图形
            plt.plot(steps_loss, losses, label='Training Loss', alpha=0.8) # 绘制 loss 曲线
            plt.title("Training Loss Curve") # 设置标题
            plt.xlabel("Step")                           # 设置 X 轴标签
            plt.ylabel("Loss")                           # 设置 Y 轴标签
            plt.legend()                                 # 显示图例
            plt.grid(True, which='both', linestyle='--') # 显示网格线
            save_path = latest_log_dir / "loss_curve.png" # 定义保存路径
            plt.savefig(save_path)                       # 保存图像
            print(f"✅ 已生成 Loss 曲线图: {save_path}")
            plt.close()                                  # 关闭当前图形，释放内存

        # 绘制 mAP 曲线
        if maps_val:
            plt.figure(figsize=(12, 6))                  # 创建一个新的图形
            # MMEngine 的 val 是在每个 epoch 结束后运行，所以 x 轴可以是 epoch 序号
            val_epochs = list(range(1, len(epochs_map) + 1))
            plt.plot(val_epochs, maps_val, marker='o', linestyle='-', label='Validation mAP@.50') # 绘制 mAP 曲线
            plt.title("Validation mAP Curve") # 设置标题
            plt.xlabel("Epoch")                          # 设置 X 轴标签
            plt.ylabel("mAP@.50")                         # 设置 Y 轴标签
            plt.xticks(val_epochs)                       # 确保每个 epoch 都有刻度
            plt.legend()                                 # 显示图例
            plt.grid(True, which='both', linestyle='--') # 显示网格线
            save_path = latest_log_dir / "mAP_curve.png" # 定义保存路径
            plt.savefig(save_path)                       # 保存图像
            print(f"✅ 已生成 mAP 曲线图: {save_path}")
            plt.close()                                  # 关闭当前图形，释放内存

            # --- 生成训练总结 ---
            best_map_index = maps_val.index(max(maps_val)) # 找到最佳 mAP 的索引
            best_map_epoch = val_epochs[best_map_index]  # 找到对应的 epoch
            best_map_value = maps_val[best_map_index]      # 获取最佳 mAP 值

            summary_path = latest_log_dir / "summary.txt" # 定义总结文件路径
            with open(summary_path, "w", encoding='utf-8') as f: # 写入总结
                f.write(f"训练完成总结\n")
                f.write("="*20 + "\n")
                f.write(f"最优 mAP@.50: {best_map_value:.4f}\n")
                f.write(f"达成于 Epoch: {best_map_epoch}\n")
            print(f"✅ 已生成训练摘要: {summary_path}")

    except Exception as e:
        print(f"❌ 解析日志或绘图时发生错误: {e}")


def main():
    """主函数，用于解析命令行参数并启动训练流程"""
    parser = argparse.ArgumentParser(
        description='🚀 智能多GPU训练启动器 v2.1 (适配 MMDetection) - 支持梯度累积和透传参数',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--config',
                        default='configs/mamba_vision/temp.py',
                        help='MMDetection 模型配置文件路径。')
    parser.add_argument('--work-dir',
                        default='./work_dirs/cascade_mask_rcnn_mamba_vision_tiny_3x_coco128',
                        help='工作目录，用于存放日志、模型和结果图表。')
    parser.add_argument('--resume-from',
                        default=None,
                        help='从指定的 checkpoint 文件恢复训练，例如: path/to/your/checkpoint.pth')
    parser.add_argument('--gpus',
                        type=int,
                        default=None,
                        help='指定使用的 GPU 数量。\n默认 (不设置此参数): 使用所有可见的 GPU。')
    parser.add_argument('--batch-size', '--bs',
                        type=int,
                        default=None,
                        help='(可选) 覆盖配置文件中每个 GPU 的 batch_size。\n'
                             '如果 OOM，可尝试 --bs 1 或 --bs 2。')
    # ✅ 新增功能：梯度累积参数
    parser.add_argument('--accum-steps', '--accum',
                        type=int,
                        default=None,
                        help='(可选, 推荐!) 梯度累积步数。\n'
                             '如果 --bs 1 仍然 OOM，这是最后的解决方案。\n'
                             '例如: --bs 1 --accum 4，效果等同于 batch_size=4，但显存占用是 bs=1 的水平。')

    # ✅【核心修复】使用 parse_known_args() 来分离已知和未知参数
    # 这使得脚本可以接受任何它本身未定义的参数（如 --cfg-options），并将它们传递下去
    args, unknown_args = parser.parse_known_args()

    # --- 准备工作 ---
    requested_gpus = args.gpus if args.gpus is not None else torch.cuda.device_count()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 工作目录已准备就绪: {work_dir.resolve()}")

    # --- 启动训练与分析 ---
    start_time = time.time()
    success = train_multi_gpu(
        config_path=args.config,
        work_dir=str(work_dir),
        resume_from=args.resume_from,
        num_gpus=requested_gpus,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps, # 传递梯度累积参数
        extra_args=unknown_args  # ✅【核心修复】将所有未知的额外参数传递给训练函数
    )

    if success:
        print("\n📈 训练已完成，开始进行日志分析和图表生成...")
        analyze_log_and_plot(work_dir)
    else:
        print("\n💔 训练未成功，跳过日志分析步骤。")

    duration = time.time() - start_time
    print(f"\n⏱️  总任务耗时: {duration / 3600:.2f} 小时 ({duration:.2f} 秒)")


if __name__ == '__main__':
    main()