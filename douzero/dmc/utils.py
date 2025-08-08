import os
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    return Env(flags.objective)


def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers


def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
                legal_actions=dict(size=(T, 7, 54), dtype=torch.int8),
                legal_actions_mask=dict(size=(T, 7), dtype=torch.int8),
                legal_step_num=dict(size=(T,), dtype=torch.int8),
                reward=dict(size=(T,), dtype=torch.float32)
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:' + str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        legal_actions_buf = {p: [] for p in positions}
        legal_actions_mask_buf = {p: [] for p in positions}
        legal_step_num_buff = {p: [] for p in positions}
        reward_buff = {p: [] for p in positions}

        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                myhand = env_output['obs_x_no_action'][:54].tolist()
                obs_z_buf[position].append(env_output['obs_z'])
                # 获取合法动作并编码
                legal_actions = obs['legal_actions'][:7]
                legal_actions_vec = [_cards2tensor(action) for action in legal_actions]
                legal_actions_buf[position].append(legal_actions_vec)

                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action = _cards2tensor(action)
                obs_action_buf[position].append(obs_action)
                legal_step_num_buff[position].append(_card2num(myhand, obs_action))
                size[position] += 1
                # mask保存
                mask = [1 if action in legal_actions else 0 for action in range(10)]  # 创建合法动作的 mask
                if _action_idx <= 9:
                    mask[_action_idx] = 2
                legal_actions_mask_buf[position].append(mask)
                position, obs, env_output = env.step(action)

                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output[
                                'episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])

                    break

            # 将数据保存到buffers
            l = 0.1
            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                        buffers[p]['legal_actions'][index][t, ...] = legal_actions_buf[p][t]
                        buffers[p]['legal_actions_mask'][index][t, ...] = legal_actions_mask_buf[p][t]
                        buffers[p]['legal_step_num'][index][t, ...] = legal_step_num_buff[p][t]
                        if t == 0:
                            adv_diff = legal_step_num_buff['landlord'][t] - min_step(legal_step_num_buff['landlord_up'][t], legal_step_num_buff['landlord_down'][t])
                        else:
                            adv_diff_t = legal_step_num_buff['landlord'][t] - min_step(legal_step_num_buff['landlord_up'][t], legal_step_num_buff['landlord_down'][t])
                            adv_diff_t_1 = legal_step_num_buff['landlord'][t-1] - min_step(legal_step_num_buff['landlord_up'][t-1], legal_step_num_buff['landlord_down'][t-1])
                            adv_diff = adv_diff_t - adv_diff_t_1
                        if p is 'landlord':
                            reward_buff[p][t] = -1.0 * adv_diff * l
                        else:
                            reward_buff[p][t] = 0.5 * adv_diff * l
                        buffers[p]['reward'][index][t, ...] = reward_buff[p][t]

                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    legal_actions_buf[p] = legal_actions_buf[p][T:]
                    legal_actions_mask_buf[p] = legal_actions_mask_buf[p][T:]
                    legal_step_num_buff[p] = legal_step_num_buff[p][T:]
                    reward_buff[p] = reward_buff[p][T:]

                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix


def min_step(a,b):
    if a > b:
        return b
    else:
        return a

def _card2num(myhand, action):
    """
    计算出牌后剩余手牌的最少出完步数（优化版）

    Args:
        myhand: 54维数组，表示手牌
        action: 54维数组，表示出牌动作

    Returns:
        int: 最少出完剩余牌的步数
    """
    import copy
    from collections import defaultdict

    # 计算剩余手牌
    remaining_cards = [myhand[i] - action[i] for i in range(54)]

    # 如果没有剩余牌，返回0
    if sum(remaining_cards) == 0:
        return 0

    # 预计算牌的点数映射（0-12对应3-A，13-14对应大小王）
    point_mapping = []
    for i in range(13):  # 3到A
        point_mapping.extend([i] * 4)
    point_mapping.extend([13, 14])  # 小王、大王

    # 缓存已经计算过的牌型状态，避免重复计算
    memo = {}

    def get_state_key(cards):
        """将牌组状态转换为可哈希的键，用于缓存"""
        return tuple(cards)

    def count_card_types(cards):
        """统计各种牌型的数量，返回结构化数据"""
        counts = [0] * 15  # 13个点数 + 2个王
        for i in range(54):
            counts[point_mapping[i]] += cards[i]

        solo_count = 0
        pair_count = 0
        trio_count = 0
        bomb_count = 0

        # 统计普通牌
        for i in range(13):
            cnt = counts[i]
            if cnt == 1:
                solo_count += 1
            elif cnt == 2:
                pair_count += 1
            elif cnt == 3:
                trio_count += 1
            elif cnt >= 4:
                bomb_count += 1

        # 处理王牌
        joker_count = counts[13] + counts[14]
        rocket = 1 if counts[13] >= 1 and counts[14] >= 1 else 0

        return {
            'solo': solo_count,
            'pair': pair_count,
            'trio': trio_count,
            'bomb': bomb_count,
            'joker': joker_count,
            'rocket': rocket
        }

    # 预初始化动态规划矩阵，只计算一次
    dp_matrix = None

    def init_matrix():
        """初始化动态规划矩阵，只执行一次"""
        nonlocal dp_matrix
        if dp_matrix is not None:
            return dp_matrix

        max_cards = 20  # 最大牌数限制
        dp = defaultdict(lambda: float('inf'))
        dp[(0, 0, 0, 0)] = 0

        for n1 in range(max_cards + 1):
            for n2 in range(max_cards + 1):
                for n3 in range(max_cards + 1):
                    for n4 in range(max_cards + 1):
                        if n1 == 0 and n2 == 0 and n3 == 0 and n4 == 0:
                            continue

                        current = float('inf')

                        # 单牌
                        if n1 > 0:
                            current = min(current, dp[(n1 - 1, n2, n3, n4)] + 1)

                        # 对子
                        if n2 > 0:
                            current = min(current, dp[(n1, n2 - 1, n3, n4)] + 1)

                        # 三张
                        if n3 > 0:
                            current = min(current, dp[(n1, n2, n3 - 1, n4)] + 1)
                            # 三张可以拆成一张+一对
                            current = min(current, dp[(n1 + 1, n2 + 1, n3 - 1, n4)])

                        # 炸弹
                        if n4 > 0:
                            current = min(current, dp[(n1, n2, n3, n4 - 1)] + 1)
                            # 炸弹可以拆成其他组合
                            current = min(current, dp[(n1 + 1, n2 + 1, n3, n4 - 1)])

                        # 三带一
                        if n3 > 0 and n1 > 0:
                            current = min(current, dp[(n1 - 1, n2, n3 - 1, n4)] + 1)

                        # 三带二
                        if n3 > 0 and n2 > 0:
                            current = min(current, dp[(n1, n2 - 1, n3 - 1, n4)] + 1)

                        # 四带二单
                        if n4 > 0 and n1 >= 2:
                            current = min(current, dp[(n1 - 2, n2, n3, n4 - 1)] + 1)

                        # 四带二对
                        if n4 > 0 and n2 >= 2:
                            current = min(current, dp[(n1, n2 - 2, n3, n4 - 1)] + 1)

                        dp[(n1, n2, n3, n4)] = current

        dp_matrix = dp
        return dp

    def now_step(cards):
        """计算当前牌组的最少步数，使用缓存"""
        key = get_state_key(cards)
        if key in memo:
            return memo[key]

        total = sum(cards)
        if total == 0:
            return 0

        # 检查牌型
        types = count_card_types(cards)

        # 处理火箭
        if types['rocket']:
            # 打出火箭后的剩余牌
            left_cards = copy.copy(cards)
            # 移除一个小王和一个大王
            left_cards[52] -= 1
            left_cards[53] -= 1
            res = min(now_step(left_cards) + 1, now_step(left_cards) + 2)
            memo[key] = res
            return res

        # 处理单王
        if types['joker'] == 1:
            left_cards = copy.copy(cards)
            if cards[52] > 0:
                left_cards[52] -= 1
            else:
                left_cards[53] -= 1
            res = now_step(left_cards) + 1
            memo[key] = res
            return res

        # 没有王的情况
        if types['joker'] == 0:
            dp = init_matrix()
            res = dp[(types['solo'], types['pair'], types['trio'], types['bomb'])]
            memo[key] = res
            return res

        # 默认返回最大值
        memo[key] = float('inf')
        return float('inf')

    def get_chains(cards):
        """获取所有可能的顺子组合，优化生成逻辑"""
        chains = []
        counts = [0] * 13  # 13个普通牌点数的数量

        for i in range(52):
            point = point_mapping[i]
            counts[point] += cards[i]

        # 单顺（至少5张连续，最多12张）
        start = 0  # 3对应的索引
        end = 8  # J对应的索引（3到J才能形成12张顺子）
        for length in range(5, 13):  # 先按长度从小到大，便于剪枝
            for s in range(start, end - length + 2):
                valid = True
                for i in range(s, s + length):
                    if counts[i] < 1:
                        valid = False
                        break
                if valid:
                    chains.append(('solo_chain', s, length))

        # 双顺（至少3对连续，最多10对）
        start = 0  # 3对应的索引
        end = 9  # Q对应的索引
        for length in range(3, 11):
            for s in range(start, end - length + 2):
                valid = True
                for i in range(s, s + length):
                    if counts[i] < 2:
                        valid = False
                        break
                if valid:
                    chains.append(('pair_chain', s, length))

        return chains

    def dfs(step, current_ans, cards):
        """深度优先搜索最优解，优化剪枝策略"""
        # 计算当前剩余牌数，快速估计下界
        remaining = sum(cards)
        min_possible = step + (remaining + 3) // 4  # 每步至少出1张，最多出4张
        if min_possible >= current_ans:
            return current_ans

        # 计算当前步数并更新最优解
        current_step = now_step(cards)
        new_ans = min(current_ans, step + current_step)
        if new_ans == step + 1:  # 不可能再优化了
            return new_ans

        # 获取所有可能的顺子组合
        chains = get_chains(cards)

        # 按顺子长度排序，优先尝试长顺子（可能减少更多步数）
        chains.sort(key=lambda x: -x[2])

        for chain_type, start, length in chains:
            # 创建剩余牌的副本
            left_cards = copy.copy(cards)
            valid = True

            if chain_type == 'solo_chain':
                # 处理单顺
                for i in range(length):
                    point = start + i
                    # 找到该点数的一张牌并移除
                    found = False
                    for j in range(4):  # 遍历该点数的4张牌
                        idx = point * 4 + j
                        if left_cards[idx] > 0:
                            left_cards[idx] -= 1
                            found = True
                            break
                    if not found:
                        valid = False
                        break

            elif chain_type == 'pair_chain':
                # 处理双顺
                for i in range(length):
                    point = start + i
                    # 找到该点数的两张牌并移除
                    removed = 0
                    for j in range(4):
                        while left_cards[point * 4 + j] > 0 and removed < 2:
                            left_cards[point * 4 + j] -= 1
                            removed += 1
                    if removed < 2:
                        valid = False
                        break

            if valid:
                # 递归搜索，更新最优解
                new_ans = min(new_ans, dfs(step + 1, new_ans, left_cards))
                if new_ans == step + 1:  # 剪枝：不可能再优化
                    return new_ans

        return new_ans

    # 初始化DP矩阵
    init_matrix()
    # 清空缓存
    memo.clear()
    # 执行DFS搜索
    result = dfs(0, float('inf'), remaining_cards)

    return result if result != float('inf') else 0
