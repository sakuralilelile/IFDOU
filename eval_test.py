import os
import argparse
import re
import torch
from glob import glob
from douzero.evaluation.simulation import evaluate


def get_checkpoint_steps(checkpoint_path):
    """从checkpoint文件名中提取步数信息"""
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'weights_(\d+)\.ckpt', filename)
    if match:
        return int(match.group(1))
    return -1


def evaluate_checkpoints(agent_type, base_dir='new_checkpoints/douzero'):
    """评估指定类型的所有checkpoints"""
    # 构建匹配模式
    pattern = os.path.join(base_dir, f"{agent_type}_weights_*.ckpt")
    # 获取所有匹配的checkpoint文件
    checkpoints = glob(pattern)
    # 按步数排序
    checkpoints.sort(key=get_checkpoint_steps)

    if not checkpoints:
        print(f"未找到{agent_type}类型的checkpoint文件")
        return

    # 基础路径设置
    sl_base = 'save_pkt/sl'
    sl_checkpoints = {
        'landlord': os.path.join(sl_base, 'landlord.ckpt'),
        'landlord_up': os.path.join(sl_base, 'landlord_up.ckpt'),
        'landlord_down': os.path.join(sl_base, 'landlord_down.ckpt')
    }

    # 输出结果文件
    result_file = f"{agent_type}_evaluation_results.txt"

    with open(result_file, 'w') as f:
        f.write(f"{agent_type}评估结果\n")
        f.write("=" * 50 + "\n")
        f.write("步数\t地主胜率\t地主ADP\t农民胜率\t农民ADP\n")

        for checkpoint in checkpoints:
            steps = get_checkpoint_steps(checkpoint)
            if steps == -1:
                continue

            # 构建当前评估使用的checkpoint组合
            current_checkpoints = sl_checkpoints.copy()

            # 特殊处理：当评估landlord_up或landlord_down时，同时替换这两个角色的checkpoint
            if agent_type in ['landlord_up', 'landlord_down']:
                # 寻找相同步数的另一个角色的checkpoint
                counterpart_type = 'landlord_down' if agent_type == 'landlord_up' else 'landlord_up'
                counterpart_pattern = os.path.join(base_dir, f"{counterpart_type}_weights_{steps}.ckpt")
                counterpart_checkpoint = glob(counterpart_pattern)

                if len(counterpart_checkpoint) == 1:
                    # 找到匹配的checkpoint，同时替换两个角色
                    current_checkpoints[agent_type] = checkpoint
                    current_checkpoints[counterpart_type] = counterpart_checkpoint[0]
                    print(f"正在评估 {agent_type} 和 {counterpart_type} 步数: {steps}")
                    print(f"使用checkpoint: {checkpoint} 和 {counterpart_checkpoint[0]}")
                else:
                    # 未找到匹配的checkpoint，跳过当前步骤
                    print(f"警告: 未找到{counterpart_type}对应的步数为{steps}的checkpoint，跳过此评估")
                    continue
            else:
                # 评估landlord时保持原逻辑不变
                current_checkpoints[agent_type] = checkpoint
                print(f"正在评估 {agent_type} 步数: {steps}")
                print(f"使用checkpoint: {checkpoint}")

            # 执行评估
            landlord_wp, landlord_adp, farmer_wp, farmer_adp = evaluate(
                current_checkpoints['landlord'],
                current_checkpoints['landlord_up'],
                current_checkpoints['landlord_down'],
                'eval_data.pkl',
                5
            )

            # 输出结果
            result_line = f"{steps}\t{landlord_wp:.4f}\t{landlord_adp:.4f}\t{farmer_wp:.4f}\t{farmer_adp:.4f}\n"
            print(result_line)
            f.write(result_line)

    print(f"评估完成，结果已保存到 {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--agent_type', type=str, default='landlord',
                        choices=['landlord', 'landlord_up', 'landlord_down'],
                        help='要评估的agent类型')

    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    # 执行评估
    evaluate_checkpoints('landlord')