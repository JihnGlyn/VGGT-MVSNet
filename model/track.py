# 形状: [S, N, 2] = [5, 1000, 2]
# S=5: 5张图像
# N=1000: 总共1000条轨迹（所有查询帧的关键点合并）
# 2: (x, y) 像素坐标

#pred_tracks[0, :, :]  # 第0张图像中所有轨迹的位置
# 例如: [[100.5, 200.3], [150.2, 300.1], [200.0, 400.5], ...]
#      轨迹0位置    轨迹1位置    轨迹2位置

#pred_tracks[1, :, :]  # 第1张图像中所有轨迹的位置
# 例如: [[102.1, 201.8], [151.5, 299.2], [201.3, 401.0], ...]
#      轨迹0位置    轨迹1位置    轨迹2位置


# 形状: [S, N] = [5, 1000]
# 每个值表示该轨迹在该帧的可见性分数 (0-1之间)

#pred_vis_scores[0, :]  # 第0张图像中所有轨迹的可见性
# 例如: [0.95, 0.87, 0.23, 0.91, ...]
#      轨迹0  轨迹1  轨迹2  轨迹3

#pred_vis_scores[1, :]  # 第1张图像中所有轨迹的可见性
# 例如: [0.92, 0.89, 0.31, 0.88, ...]
#      轨迹0  轨迹1  轨迹2  轨迹3


# 输出结果示例
# view_matches = {
#     0: [(632, 1), (631, 2), (626, 4), (623, 3)],
#     1: [(654, 2), (643, 3), (640, 4), (632, 0)],
#     2: [(654, 1), (647, 4), (631, 0), (630, 3)],
#     3: [(644, 4), (643, 1), (630, 2), (623, 0)],
#     4: [(647, 2), (644, 3), (640, 1), (626, 0)]
# }

import cv2
from itertools import combinations
import numpy as np


def count_matches_between_pairs(pred_tracks, pred_vis_scores, vis_thresh=0.2, use_ransac=False, top_k=10):
    """
    统计两两视图之间的特征匹配点数并排序，输出每个视角与匹配度最高的前k个其他视角
    
    Args:
        pred_tracks: [S, N, 2] 轨迹坐标
        pred_vis_scores: [S, N] 可见性分数
        vis_thresh: 可见性阈值，默认0.2
        use_ransac: 是否使用RANSAC过滤，默认False（仅计数）
        top_k: 返回前k个最佳匹配，默认10
    
    Returns:
        dict: 每个视角的匹配结果
        {
            view_i: [
                (match_score, matched_view_j),
                (match_score, matched_view_j),
                ...
            ]
        }
    """
    
    S, N = pred_vis_scores.shape
    results = []
    
    # 计算所有图像对之间的匹配点数
    for i, j in combinations(range(S), 2):
        # 找到在两帧都可见的轨迹
        both_visible = (pred_vis_scores[i] > vis_thresh) & (pred_vis_scores[j] > vis_thresh)
        
        if not use_ransac:
            # 直接计数
            count = int(both_visible.sum())
        else:
            # 使用RANSAC过滤
            if both_visible.sum() < 8:
                count = 0
            else:
                pts_i = pred_tracks[i, both_visible, :].astype(np.float32)
                pts_j = pred_tracks[j, both_visible, :].astype(np.float32)
                
                F, inliers = cv2.findFundamentalMat(
                    pts_i, pts_j,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0,
                    confidence=0.999,
                    maxIters=10000
                )
                count = int(inliers.sum()) if inliers is not None else 0
        
        results.append((count, i, j))
    
    # 为每个视角构建匹配字典
    view_matches = {}
    for i in range(S):
        view_matches[i] = []
    
    # 填充匹配结果（双向）
    for count, i, j in results:
        view_matches[i].append((count, j))
        view_matches[j].append((count, i))
    
    # 对每个视角的匹配结果按匹配点数降序排序，并取前top_k个
    for view_idx in view_matches:
        view_matches[view_idx].sort(key=lambda x: x[0], reverse=True)
        view_matches[view_idx] = view_matches[view_idx][:top_k]
    
    return view_matches


def print_view_matches(view_matches):
    """
    打印每个视角的匹配结果
    
    Args:
        view_matches: count_matches_between_pairs函数的返回结果
    """
    for view_idx in sorted(view_matches.keys()):
        matches = view_matches[view_idx]
        print(f"\n视角 {view_idx} 的匹配结果:")
        print("-" * 40)
        
        if not matches:
            print("  无匹配的视角")
            continue
            
        for rank, (match_score, matched_view) in enumerate(matches, 1):
            print(f"  第{rank:2d}名: 视角{matched_view:2d} - 匹配点数: {match_score:4d}")

# 视角 47 的匹配结果:
# ----------------------------------------
#   第 1名: 视角32 - 匹配点数: 3168
#   第 2名: 视角24 - 匹配点数: 3142
#   第 3名: 视角21 - 匹配点数: 3111
#   第 4名: 视角23 - 匹配点数: 3010
#   第 5名: 视角13 - 匹配点数: 3002
#   第 6名: 视角18 - 匹配点数: 2979
#   第 7名: 视角16 - 匹配点数: 2942
#   第 8名: 视角 3 - 匹配点数: 2855
#   第 9名: 视角43 - 匹配点数: 2845
#   第10名: 视角44 - 匹配点数: 2833

# 视角 48 的匹配结果:
# ----------------------------------------
#   第 1名: 视角41 - 匹配点数: 2122
#   第 2名: 视角37 - 匹配点数: 2100
#   第 3名: 视角26 - 匹配点数: 2092
#   第 4名: 视角42 - 匹配点数: 2063
#   第 5名: 视角39 - 匹配点数: 2061
#   第 6名: 视角14 - 匹配点数: 2042
#   第 7名: 视角45 - 匹配点数: 2041
#   第 8名: 视角38 - 匹配点数: 2013
#   第 9名: 视角11 - 匹配点数: 2009
#   第10名: 视角31 - 匹配点数: 1997