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
                legal_step_num_buff[position].append(_card_num(myhand, obs_action))
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
                        if p == 'landlord':
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

def _card_num(myhand, action):
    """
    计算在出了 action 之后，剩余的卡牌最少需要多少步出完。
    加入了对三带一、三带一对和四带二的处理。

    参数:
    myhand (list): 54维列表，表示当前手牌。myhand[i] = 1 表示有牌，0 表示无。
    action (list): 54维列表，表示本次出的牌。action[i] = 1 表示出了牌，0 表示无。

    返回:
    int: 剩余牌需要的最少出牌步数 (基于贪心策略)。
    """

    # --- Step 1: 计算剩余的精确卡牌 (54维) ---
    current_specific_cards = list(myhand)
    for i in range(54):
        # 确保 action 是合法的，即 action[i] <= myhand[i]
        if action[i] > current_specific_cards[i]:
            # 如果动作不合法，可以抛出错误或返回特殊值
            return -1  # 表示错误或非法操作

        current_specific_cards[i] -= action[i]

    steps = 0

    # --- Step 2: 辅助函数 - 从精确卡牌更新每种牌点的数量 ---
    # rank_counts[0] for '3's, ..., rank_counts[11] for 'A's, rank_counts[12] for '2's,
    # rank_counts[13] for 小王, rank_counts[14] for 大王。
    def _update_rank_counts_from_specific(specific_cards):
        counts = [0] * 15
        for j in range(52):  # 标准52张牌 (索引 0-51)
            if specific_cards[j] == 1:
                counts[j // 4] += 1
        if specific_cards[52] == 1: counts[13] += 1  # 小王
        if specific_cards[53] == 1: counts[14] += 1  # 大王
        return counts

    # --- Step 3: 处理火箭 (双王) ---
    # 注意：火箭在任何情况下都是一步且优先级最高
    if current_specific_cards[52] == 1 and current_specific_cards[53] == 1:
        steps += 1
        current_specific_cards[52] = 0
        current_specific_cards[53] = 0

    # --- Step 4: 循环迭代处理所有可能的牌型，直到没有牌剩余 ---
    # 每次循环尝试打出优先级最高的牌型，并更新剩余牌数。
    # 这个循环在最坏情况下只会执行固定的小次数 (例如，最大手牌数 / 1)。
    while True:
        # 在每次迭代开始时，重新获取当前牌点数量的快照
        remaining_rank_counts = _update_rank_counts_from_specific(current_specific_cards)

        # 检查是否还有牌剩余
        if sum(remaining_rank_counts) == 0:
            break  # 所有牌已出完

        # 记录本轮是否成功打出牌型，如果没有，则说明无法再组织任何牌型，进入单牌计数
        progress_made_in_this_iteration = False

        # --- 获取当前可用的单张、对子、三张、四张牌的牌点列表 (用于“带”牌型) ---
        # 排除 2 和大小王参与顺子/连对，但可以被带或作为单张。
        singles_available = []  # 牌点，拥有1张牌
        pairs_available = []  # 牌点，拥有2张牌
        triples_available = []  # 牌点，拥有3张牌
        quads_available = []  # 牌点，拥有4张牌

        # 遍历标准牌点 (3-2, 索引 0-12)
        for i in range(13):
            if remaining_rank_counts[i] == 1:
                singles_available.append(i)
            elif remaining_rank_counts[i] == 2:
                pairs_available.append(i)
            elif remaining_rank_counts[i] == 3:
                triples_available.append(i)
            elif remaining_rank_counts[i] == 4:
                quads_available.append(i)

        # 加上大小王作为单张，如果它们未被用作火箭
        if remaining_rank_counts[13] == 1: singles_available.append(13)
        if remaining_rank_counts[14] == 1: singles_available.append(14)

        # 为了在“带”牌型中优先带最小的牌，对可用牌点进行排序
        singles_available.sort()
        pairs_available.sort()

        # --- 优先处理长的、连续的牌型 (顺子、连对、飞机) ---
        # 遍历从最长到最短，确保优先打出大牌型

        # 4.1 单张顺子 (至少5张，3-A，索引 0-11)
        # 从12张牌长的顺子 (3-A) 往下找，直到5张
        found_pattern_this_sub_phase = False
        for length in range(12, 4, -1):  # 长度从12递减到5
            for start_rank_idx in range(0, 12 - length + 1):  # 遍历可能的起始点
                is_straight = True
                for k in range(length):
                    if remaining_rank_counts[start_rank_idx + k] < 1:
                        is_straight = False
                        break
                if is_straight:
                    steps += 1
                    for k in range(length):
                        remaining_rank_counts[start_rank_idx + k] -= 1
                        # 还需要从 specific_cards 中移除
                        for card_idx_offset in range(4):  # 移除该牌点对应的任意一张花色
                            specific_card_idx = (start_rank_idx * 4) + card_idx_offset
                            if current_specific_cards[specific_card_idx] == 1:
                                current_specific_cards[specific_card_idx] = 0
                                break
                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break  # 找到一个顺子，重新评估所有牌型
            if found_pattern_this_sub_phase: break
        if found_pattern_this_sub_phase: continue  # 有进展，重新开始大循环

        # 4.2 三张连对 (飞机，至少2组，3-A，索引 0-11)
        found_pattern_this_sub_phase = False
        for length in range(10, 1, -1):  # 长度从10递减到2
            for start_rank_idx in range(0, 12 - length + 1):
                is_triple_seq = True
                for k in range(length):
                    if remaining_rank_counts[start_rank_idx + k] < 3:
                        is_triple_seq = False
                        break
                if is_triple_seq:
                    steps += 1
                    for k in range(length):
                        remaining_rank_counts[start_rank_idx + k] -= 3
                        # 从 specific_cards 移除三张
                        for card_idx_offset in range(4):  # 移除该牌点对应的三张花色
                            specific_card_idx = (start_rank_idx * 4) + card_idx_offset
                            if current_specific_cards[specific_card_idx] == 1:
                                current_specific_cards[specific_card_idx] = 0
                                # 优化: 确保移除3张
                                # For perfect specific card removal, we need to handle this more carefully.
                                # For O(1) greedy, `rank_counts` is sufficient to track available sets.
                                # Let's keep `current_specific_cards` updated by rank index.
                                for _ in range(3):
                                    specific_card_base = (start_rank_idx + k) * 4
                                    for idx_offset in range(4):
                                        if current_specific_cards[specific_card_base + idx_offset] == 1:
                                            current_specific_cards[specific_card_base + idx_offset] = 0
                                            break
                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break
            if found_pattern_this_sub_phase: break
        if found_pattern_this_sub_phase: continue

        # 4.3 双张连对 (连对，至少2组，3-A，索引 0-11)
        found_pattern_this_sub_phase = False
        for length in range(10, 1, -1):  # 长度从10递减到2
            for start_rank_idx in range(0, 12 - length + 1):
                is_pair_seq = True
                for k in range(length):
                    if remaining_rank_counts[start_rank_idx + k] < 2:
                        is_pair_seq = False
                        break
                if is_pair_seq:
                    steps += 1
                    for k in range(length):
                        remaining_rank_counts[start_rank_idx + k] -= 2
                        # 从 specific_cards 移除两张
                        for _ in range(2):
                            specific_card_base = (start_rank_idx + k) * 4
                            for idx_offset in range(4):
                                if current_specific_cards[specific_card_base + idx_offset] == 1:
                                    current_specific_cards[specific_card_base + idx_offset] = 0
                                    break
                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break
            if found_pattern_this_sub_phase: break
        if found_pattern_this_sub_phase: continue

        # --- 优先级：四带二 (四张同点 + 两张单牌 或 四张同点 + 两对牌) ---
        found_pattern_this_sub_phase = False
        # 遍历四张牌组 (从最小的牌点开始尝试带走小牌)
        for quad_rank_idx in sorted(quads_available):
            if remaining_rank_counts[quad_rank_idx] >= 4:  # 确保牌还在

                # 尝试带两张单牌
                taken_singles = []
                for s_rank in singles_available:
                    # 确保不是带自己的牌，且该单张还在
                    if s_rank != quad_rank_idx and remaining_rank_counts[s_rank] >= 1:
                        taken_singles.append(s_rank)
                        if len(taken_singles) == 2: break

                if len(taken_singles) == 2:
                    steps += 1
                    remaining_rank_counts[quad_rank_idx] -= 4
                    for s_rank in taken_singles:
                        remaining_rank_counts[s_rank] -= 1

                    # 从 specific_cards 移除
                    for _ in range(4):  # 移除四张牌
                        specific_card_base = quad_rank_idx * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break
                    for s_rank in taken_singles:  # 移除两张单牌
                        # 小王大王特殊处理
                        if s_rank == 13:
                            current_specific_cards[52] = 0
                        elif s_rank == 14:
                            current_specific_cards[53] = 0
                        else:
                            specific_card_base = s_rank * 4
                            for idx_offset in range(4):
                                if current_specific_cards[specific_card_base + idx_offset] == 1:
                                    current_specific_cards[specific_card_base + idx_offset] = 0
                                    break

                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break  # 找到四带二，重新评估

                # 如果不能带两张单牌，尝试带两对牌
                taken_pairs = []
                for p_rank in pairs_available:
                    # 确保不是带自己的牌，且该对子还在
                    if p_rank != quad_rank_idx and remaining_rank_counts[p_rank] >= 2:
                        taken_pairs.append(p_rank)
                        if len(taken_pairs) == 2: break

                if len(taken_pairs) == 2:
                    steps += 1
                    remaining_rank_counts[quad_rank_idx] -= 4
                    for p_rank in taken_pairs:
                        remaining_rank_counts[p_rank] -= 2

                    # 从 specific_cards 移除
                    for _ in range(4):  # 移除四张牌
                        specific_card_base = quad_rank_idx * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break
                    for p_rank in taken_pairs:  # 移除两对牌
                        for _ in range(2):
                            specific_card_base = p_rank * 4
                            for idx_offset in range(4):
                                if current_specific_cards[specific_card_base + idx_offset] == 1:
                                    current_specific_cards[specific_card_base + idx_offset] = 0
                                    break

                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break  # 找到四带二，重新评估

        if found_pattern_this_sub_phase: continue  # 有进展，重新开始大循环

        # --- 优先级：三带一 / 三带一对 ---
        found_pattern_this_sub_phase = False
        # 遍历三张牌组 (从最小的牌点开始尝试带走小牌)
        for triple_rank_idx in sorted(triples_available):
            if remaining_rank_counts[triple_rank_idx] >= 3:  # 确保牌还在

                # 尝试带一张单牌
                found_single_to_take = None
                for s_rank in singles_available:
                    # 确保不是带自己的牌，且该单张还在
                    if s_rank != triple_rank_idx and remaining_rank_counts[s_rank] >= 1:
                        found_single_to_take = s_rank
                        break

                if found_single_to_take is not None:
                    steps += 1
                    remaining_rank_counts[triple_rank_idx] -= 3
                    remaining_rank_counts[found_single_to_take] -= 1

                    # 从 specific_cards 移除
                    for _ in range(3):  # 移除三张牌
                        specific_card_base = triple_rank_idx * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break
                    # 移除一张单牌
                    if found_single_to_take == 13:
                        current_specific_cards[52] = 0
                    elif found_single_to_take == 14:
                        current_specific_cards[53] = 0
                    else:
                        specific_card_base = found_single_to_take * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break

                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break  # 找到三带一，重新评估

                # 如果不能带单牌，尝试带一对牌
                found_pair_to_take = None
                for p_rank in pairs_available:
                    # 确保不是带自己的牌，且该对子还在
                    if p_rank != triple_rank_idx and remaining_rank_counts[p_rank] >= 2:
                        found_pair_to_take = p_rank
                        break

                if found_pair_to_take is not None:
                    steps += 1
                    remaining_rank_counts[triple_rank_idx] -= 3
                    remaining_rank_counts[found_pair_to_take] -= 2

                    # 从 specific_cards 移除
                    for _ in range(3):  # 移除三张牌
                        specific_card_base = triple_rank_idx * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break
                    for _ in range(2):  # 移除一对牌
                        specific_card_base = found_pair_to_take * 4
                        for idx_offset in range(4):
                            if current_specific_cards[specific_card_base + idx_offset] == 1:
                                current_specific_cards[specific_card_base + idx_offset] = 0
                                break

                    progress_made_in_this_iteration = True
                    found_pattern_this_sub_phase = True
                    break  # 找到三带一对，重新评估

        if found_pattern_this_sub_phase: continue  # 有进展，重新开始大循环

        # --- 处理纯牌型 (炸弹、三张、对子、单张) ---
        # 这些牌型优先级低于带牌型和连贯牌型，但高于单个散牌。

        # 4.4 纯炸弹 (四张同点，3-2，索引 0-12)
        found_pattern_this_sub_phase = False
        for i in range(13):
            if remaining_rank_counts[i] >= 4:
                steps += 1
                remaining_rank_counts[i] -= 4
                # 从 specific_cards 移除四张
                for _ in range(4):
                    specific_card_base = i * 4
                    for idx_offset in range(4):
                        if current_specific_cards[specific_card_base + idx_offset] == 1:
                            current_specific_cards[specific_card_base + idx_offset] = 0
                            break
                progress_made_in_this_iteration = True
                found_pattern_this_sub_phase = True
                break
        if found_pattern_this_sub_phase: continue

        # 4.5 纯三张 (三张同点，3-2，索引 0-12)
        found_pattern_this_sub_phase = False
        for i in range(13):
            if remaining_rank_counts[i] >= 3:
                steps += 1
                remaining_rank_counts[i] -= 3
                # 从 specific_cards 移除三张
                for _ in range(3):
                    specific_card_base = i * 4
                    for idx_offset in range(4):
                        if current_specific_cards[specific_card_base + idx_offset] == 1:
                            current_specific_cards[specific_card_base + idx_offset] = 0
                            break
                progress_made_in_this_iteration = True
                found_pattern_this_sub_phase = True
                break
        if found_pattern_this_sub_phase: continue

        # 4.6 纯对子 (两张同点，3-2，索引 0-12)
        found_pattern_this_sub_phase = False
        for i in range(13):
            if remaining_rank_counts[i] >= 2:
                steps += 1
                remaining_rank_counts[i] -= 2
                # 从 specific_cards 移除两张
                for _ in range(2):
                    specific_card_base = i * 4
                    for idx_offset in range(4):
                        if current_specific_cards[specific_card_base + idx_offset] == 1:
                            current_specific_cards[specific_card_base + idx_offset] = 0
                            break
                progress_made_in_this_iteration = True
                found_pattern_this_sub_phase = True
                break
        if found_pattern_this_sub_phase: continue

        # 4.7 纯单张 (包括大小王，索引 0-14)
        found_pattern_this_sub_phase = False
        for i in range(15):
            if remaining_rank_counts[i] >= 1:
                steps += 1
                remaining_rank_counts[i] -= 1
                # 从 specific_cards 移除一张
                if i == 13:
                    current_specific_cards[52] = 0  # 小王
                elif i == 14:
                    current_specific_cards[53] = 0  # 大王
                else:  # 标准牌
                    specific_card_base = i * 4
                    for idx_offset in range(4):
                        if current_specific_cards[specific_card_base + idx_offset] == 1:
                            current_specific_cards[specific_card_base + idx_offset] = 0
                            break
                progress_made_in_this_iteration = True
                found_pattern_this_sub_phase = True
                break
        if found_pattern_this_sub_phase: continue

        # 如果本轮没有任何牌型可以打出，则跳出循环
        if not progress_made_in_this_iteration:
            break

    return steps
