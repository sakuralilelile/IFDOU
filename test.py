def _card_num(myhand, action):
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


# 测试代码
if __name__ == "__main__":
    # 手牌有3和5
    myhand = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 出牌为5
    action = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    result = _card_num(myhand, action)
    print(f"剩余牌最少需要 {result} 步出完")
