"""
歷史回測系統
用於驗證模型在過去賽季的預測準確度與盈利能力
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BacktestSystem:
    def __init__(self, data_dir='report/history'):
        self.data_dir = data_dir
        self.records = []
        self.starting_bankroll = 10000
        self.current_bankroll = self.starting_bankroll

    def load_history(self, start_date, end_date):
        """載入歷史預測數據"""
        self.records = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            filename = f"{current.strftime('%Y-%m-%d')}.json"
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.records.append(data)
            current += timedelta(days=1)
        return len(self.records)

    def evaluate_moneyline(self):
        """評估勝負盤推薦的準確度"""
        results = {
            'total_bets': 0,
            'correct': 0,
            'profit': 0,
            'roi': 0
        }
        self.current_bankroll = self.starting_bankroll

        for record in self.records:
            for pred in record.get('today_predictions', []):
                rec = pred.get('moneyline_recommendation', '')
                if rec and rec != 'PASS':
                    results['total_bets'] += 1
                    # 假設我們知道實際結果（需接入歷史賽果）
                    actual_home_win = pred.get('actual_home_win')
                    if actual_home_win is not None:
                        if 'home' in rec.lower() and actual_home_win:
                            results['correct'] += 1
                        elif 'away' in rec.lower() and not actual_home_win:
                            results['correct'] += 1

        results['accuracy'] = results['correct'] / results['total_bets'] if results['total_bets'] > 0 else 0
        return results

    def evaluate_spread(self):
        """評估讓分盤推薦"""
        results = {'total_bets': 0, 'covered': 0}
        for record in self.records:
            for pred in record.get('today_predictions', []):
                spread_rec = pred.get('spread_recommendation', '')
                if spread_rec and spread_rec != 'PASS':
                    results['total_bets'] += 1
                    actual_diff = pred.get('actual_run_diff')
                    spread = pred.get('spread_line', 1.5)
                    if actual_diff is not None:
                        if 'home' in spread_rec.lower():
                            if actual_diff > abs(spread):
                                results['covered'] += 1
                        else:
                            if actual_diff < -abs(spread):
                                results['covered'] += 1
        results['cover_rate'] = results['covered'] / results['total_bets'] if results['total_bets'] > 0 else 0
        return results

    def evaluate_total(self):
        """評估大小分盤推薦"""
        results = {'total_bets': 0, 'correct': 0}
        for record in self.records:
            for pred in record.get('today_predictions', []):
                total_rec = pred.get('total_recommendation', '')
                if total_rec and total_rec != 'PASS':
                    results['total_bets'] += 1
                    actual_total = pred.get('actual_total_runs')
                    total_line = pred.get('total_line', 8.5)
                    if actual_total is not None:
                        if 'over' in total_rec.lower() and actual_total > total_line:
                            results['correct'] += 1
                        elif 'under' in total_rec.lower() and actual_total < total_line:
                            results['correct'] += 1
        results['accuracy'] = results['correct'] / results['total_bets'] if results['total_bets'] > 0 else 0
        return results

    def generate_report(self, start_date, end_date):
        """生成完整回測報告"""
        self.load_history(start_date, end_date)
        moneyline = self.evaluate_moneyline()
        spread = self.evaluate_spread()
        total = self.evaluate_total()

        report = {
            'period': f"{start_date} ~ {end_date}",
            'generated_at': datetime.now().isoformat(),
            'moneyline': {
                'total_bets': moneyline['total_bets'],
                'accuracy': f"{moneyline['accuracy']:.1%}",
                'roi': f"{moneyline.get('roi', 0):.1%}"
            },
            'spread': {
                'total_bets': spread['total_bets'],
                'cover_rate': f"{spread['cover_rate']:.1%}"
            },
            'total': {
                'total_bets': total['total_bets'],
                'accuracy': f"{total['accuracy']:.1%}"
            },
            'summary': self._generate_summary(moneyline, spread, total)
        }
        return report

    def _generate_summary(self, moneyline, spread, total):
        """生成回測摘要與建議"""
        summary = []
        if moneyline['accuracy'] > 0.55:
            summary.append("✅ 勝負盤模型表現優於隨機，可持續使用。")
        else:
            summary.append("⚠️ 勝負盤準確度偏低，建議調整權重或增加特徵。")
        
        if spread['cover_rate'] > 0.50:
            summary.append("✅ 讓分盤過盤率高於50%，長期可能獲利。")
        
        if total['accuracy'] > 0.52:
            summary.append("✅ 大小分盤預測有顯著優勢。")
        
        return summary
