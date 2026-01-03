"""
排行榜系统 - Leaderboard
记录和展示用户提交结果
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from datetime import datetime
import json
import os

from .challenge_base import ChallengeResult


class Leaderboard:
    """
    排行榜管理系统

    功能:
    - 记录用户提交
    - 排名
    - 可视化对比
    """

    def __init__(self, challenge_name: str, storage_dir: str = "./challenge_submissions"):
        """
        Parameters
        ----------
        challenge_name : str
            挑战名称
        storage_dir : str
            提交记录存储目录
        """
        self.challenge_name = challenge_name
        self.storage_dir = storage_dir
        self.storage_file = os.path.join(storage_dir, f"{challenge_name.replace(' ', '_')}.json")

        # 创建存储目录
        os.makedirs(storage_dir, exist_ok=True)

        # 加载历史记录
        self.submissions = self.load_submissions()

    def add_submission(self, result: ChallengeResult):
        """添加新提交"""
        submission = {
            'user_name': result.user_name,
            'submission_time': result.submission_time,
            'score': result.score,
            'primary_metric': result.primary_metric,
            'secondary_metrics': result.secondary_metrics,
            'method_description': result.method_description,
            'execution_time': result.execution_time
        }

        self.submissions.append(submission)
        self.save_submissions()

    def get_rankings(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取排名

        Parameters
        ----------
        top_n : int, optional
            返回前 N 名

        Returns
        -------
        rankings : pd.DataFrame
            排名表
        """
        if not self.submissions:
            return pd.DataFrame()

        df = pd.DataFrame(self.submissions)

        # 按得分降序排序
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1

        if top_n is not None:
            df = df.head(top_n)

        return df

    def get_user_history(self, user_name: str) -> pd.DataFrame:
        """获取用户历史提交"""
        if not self.submissions:
            return pd.DataFrame()

        user_submissions = [s for s in self.submissions if s['user_name'] == user_name]

        if not user_submissions:
            return pd.DataFrame()

        df = pd.DataFrame(user_submissions)
        df = df.sort_values('submission_time').reset_index(drop=True)

        return df

    def plot_rankings(self, top_n: int = 10) -> go.Figure:
        """
        可视化排名

        Parameters
        ----------
        top_n : int
            展示前 N 名

        Returns
        -------
        fig : go.Figure
            Plotly 图表
        """
        rankings = self.get_rankings(top_n=top_n)

        if rankings.empty:
            # 返回空图表
            fig = go.Figure()
            fig.add_annotation(
                text="No submissions yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Scores', 'Primary Metric Distribution'),
            specs=[[{"type": "bar"}, {"type": "box"}]]
        )

        # 1. Top scores
        colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#2D9CDB'
                  for i in range(len(rankings))]

        fig.add_trace(go.Bar(
            x=rankings['user_name'],
            y=rankings['score'],
            marker_color=colors,
            text=rankings['score'].round(2),
            textposition='outside',
            name='Score'
        ), row=1, col=1)

        # 2. Primary metric distribution
        fig.add_trace(go.Box(
            y=rankings['primary_metric'],
            name='Primary Metric',
            marker_color='#27AE60'
        ), row=1, col=2)

        fig.update_layout(
            height=400,
            template='plotly_white',
            showlegend=False,
            title_text=f'{self.challenge_name} - Leaderboard'
        )

        fig.update_xaxes(title_text='User', row=1, col=1)
        fig.update_yaxes(title_text='Score (0-100)', row=1, col=1)
        fig.update_yaxes(title_text='Primary Metric', row=1, col=2)

        return fig

    def plot_user_progress(self, user_name: str) -> go.Figure:
        """
        可视化用户进步曲线

        Parameters
        ----------
        user_name : str
            用户名

        Returns
        -------
        fig : go.Figure
            Plotly 图表
        """
        history = self.get_user_history(user_name)

        if history.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No submissions from {user_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig

        fig = go.Figure()

        # Score 曲线
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history['score'],
            mode='lines+markers',
            name='Score',
            line=dict(color='#2D9CDB', width=2),
            marker=dict(size=8)
        ))

        # 最佳得分线
        best_score = history['score'].max()
        fig.add_hline(
            y=best_score,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Best: {best_score:.2f}"
        )

        fig.update_layout(
            title=f"{user_name}'s Progress",
            xaxis_title='Attempt',
            yaxis_title='Score (0-100)',
            template='plotly_white',
            height=400
        )

        return fig

    def compare_methods(self, top_n: int = 5) -> go.Figure:
        """
        对比不同方法的性能

        Parameters
        ----------
        top_n : int
            对比前 N 个提交

        Returns
        -------
        fig : go.Figure
            Plotly 图表
        """
        rankings = self.get_rankings(top_n=top_n)

        if rankings.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No submissions yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig

        # 提取次要指标
        secondary_keys = list(rankings.iloc[0]['secondary_metrics'].keys())

        # 选择几个关键指标展示
        key_metrics = secondary_keys[:4] if len(secondary_keys) > 4 else secondary_keys

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in key_metrics]
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for idx, metric in enumerate(key_metrics):
            row, col = positions[idx]

            values = [s['secondary_metrics'].get(metric, 0) for s in self.submissions[:top_n]]
            users = rankings['user_name'].tolist()

            fig.add_trace(go.Bar(
                x=users,
                y=values,
                name=metric,
                marker_color='#2D9CDB',
                showlegend=False
            ), row=row, col=col)

        fig.update_layout(
            height=600,
            template='plotly_white',
            title_text='Method Comparison - Key Metrics'
        )

        return fig

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns
        -------
        stats : dict
            统计数据
        """
        if not self.submissions:
            return {
                'total_submissions': 0,
                'unique_users': 0,
                'avg_score': 0,
                'best_score': 0,
                'latest_submission': None
            }

        df = pd.DataFrame(self.submissions)

        stats = {
            'total_submissions': len(self.submissions),
            'unique_users': df['user_name'].nunique(),
            'avg_score': df['score'].mean(),
            'best_score': df['score'].max(),
            'latest_submission': df['submission_time'].max()
        }

        return stats

    def save_submissions(self):
        """保存提交记录到文件"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.submissions, f, indent=2)

    def load_submissions(self) -> List[Dict]:
        """从文件加载提交记录"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load submissions: {e}")
                return []
        return []

    def clear_submissions(self):
        """清空所有提交记录"""
        self.submissions = []
        self.save_submissions()

    def export_to_csv(self, filepath: str):
        """
        导出排名到 CSV

        Parameters
        ----------
        filepath : str
            导出路径
        """
        rankings = self.get_rankings()

        if not rankings.empty:
            # 展平 secondary_metrics
            secondary_df = pd.json_normalize(rankings['secondary_metrics'])
            export_df = pd.concat([
                rankings.drop('secondary_metrics', axis=1),
                secondary_df
            ], axis=1)

            export_df.to_csv(filepath, index=False)

    def get_leaderboard_markdown(self, top_n: int = 10) -> str:
        """
        生成 Markdown 格式的排行榜

        Parameters
        ----------
        top_n : int
            显示前 N 名

        Returns
        -------
        markdown : str
            Markdown 文本
        """
        rankings = self.get_rankings(top_n=top_n)

        if rankings.empty:
            return "### No submissions yet\n\nBe the first to submit!"

        # 构建表格
        md = f"### {self.challenge_name} - Top {top_n} Leaderboard\n\n"

        # 表头
        md += "| Rank | User | Score | Primary Metric | Submission Time |\n"
        md += "|------|------|-------|----------------|------------------|\n"

        # 数据行
        for _, row in rankings.iterrows():
            rank_icon = "🥇" if row['rank'] == 1 else "🥈" if row['rank'] == 2 else "🥉" if row['rank'] == 3 else ""
            md += f"| {row['rank']} {rank_icon} | {row['user_name']} | {row['score']:.2f} | {row['primary_metric']:.4f} | {row['submission_time']} |\n"

        # 统计信息
        stats = self.get_statistics()
        md += f"\n**Statistics**: {stats['total_submissions']} submissions from {stats['unique_users']} users\n"

        return md
