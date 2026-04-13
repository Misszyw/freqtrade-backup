# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
Titan V4 缠论增强版 - SOL合约策略
结合泰坦之怒V4.0共振系统 + SOL实战优化

核心优势结合:
1. 共振评分系统 - 多指标确认
2. 1:2盈亏比 - 止损-5%, 止盈+10%
3. 背驰检测 - MACD顶底背驰
4. 二分法买入 - 趋势回调买入，不追高
5. EMA趋势过滤 - 大于EMA200才做多

合约设置:
- 杠杆: 5x
- 周期: 4小时
- 交易对: SOL/USDT
- 风险: 每笔最大2%
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class TitanV4Strategy(IStrategy):
    """
    Titan V4 缠论增强版 - SOL合约策略
    
    策略核心:
    - 只交易SOL/USDT合约
    - 4小时周期
    - 共振评分 >= 5 分才入场
    - 止损: -5%
    - 止盈: +10% (1:2盈亏比)
    - 追踪止损: 2%
    """
    
    # ========== 策略元数据 ==========
    timeframe = "4h"
    leverage = 5  # 5倍杠杆
    
    # 最小共振分数
    min_resonance_score = 5
    
    # ========== 盈亏设置 ==========
    # 止损 -5%
    stoploss = -0.05
    
    # 止盈 +10%
    minimal_roi = {
        "0": 0.10
    }
    
    # 追踪止损
    use_trailing_stop = True
    trailing_stop = True
    trailing_stop_positive = 0.015  # 盈利1.5%后启动
    trailing_stop_positive_offset = 0.02  # 追踪距离2%
    trailing_only_offset_is_reached = True
    
    # ========== 指标参数 ==========
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    ema_short_period = 9
    ema_medium_period = 20
    ema_long_period = 50
    ema_trend_period = 200
    bb_period = 20
    bb_std = 2
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算所有技术指标
        """
        # ---------- RSI ----------
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        
        # ---------- MACD ----------
        macd = ta.MACD(dataframe, 
                       fastperiod=self.macd_fast, 
                       slowperiod=self.macd_slow, 
                       signalperiod=self.macd_signal)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['macd_hist'] = dataframe['macd'] - dataframe['macdsignal']
        
        # ---------- EMA均线 ----------
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=self.ema_short_period)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=self.ema_medium_period)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=self.ema_long_period)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=self.ema_trend_period)
        
        # ---------- 布林带 ----------
        bollinger = qtpylib.bollinger_bands(
            dataframe['close'], 
            window=self.bb_period, 
            stds=self.bb_std
        )
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_middle'] = bollinger['middle']
        dataframe['bb_lower'] = bollinger['lower']
        
        # ---------- 成交量 ----------
        dataframe['volume_ma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # ---------- 趋势判断 ----------
        # 大趋势: 价格 > EMA200
        dataframe['trend_up'] = dataframe['close'] > dataframe['ema_200']
        dataframe['trend_down'] = dataframe['close'] < dataframe['ema_200']
        
        # ---------- 交叉信号 ----------
        # EMA金叉/死叉
        dataframe['ema_cross_up'] = (
            (dataframe['ema_9'] > dataframe['ema_20']) & 
            (dataframe['ema_9'].shift(1) <= dataframe['ema_20'].shift(1))
        )
        dataframe['ema_cross_down'] = (
            (dataframe['ema_9'] < dataframe['ema_20']) & 
            (dataframe['ema_9'].shift(1) >= dataframe['ema_20'].shift(1))
        )
        
        # MACD金叉/死叉
        dataframe['macd_cross_up'] = (
            (dataframe['macd'] > dataframe['macdsignal']) & 
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        )
        dataframe['macd_cross_down'] = (
            (dataframe['macd'] < dataframe['macdsignal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
        )
        
        # ---------- 背驰检测 ----------
        # 底背驰: 价格创新低，但MACD柱状图没有创新低
        dataframe['price_new_low'] = dataframe['close'] == dataframe['close'].rolling(20).min()
        dataframe['bullish_divergence'] = (
            dataframe['price_new_low'] & 
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(5))
        )
        
        # 顶背驰: 价格创新高，但MACD柱状图没有创新高
        dataframe['price_new_high'] = dataframe['close'] == dataframe['close'].rolling(20).max()
        dataframe['bearish_divergence'] = (
            dataframe['price_new_high'] & 
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(5))
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号 - 二分法（回调买入）
        
        只在回调时买入，不追高
        """
        # 只交易SOL/USDT
        if metadata['pair'] != 'SOL/USDT':
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            return dataframe
        
        # ---------- 做多条件 ----------
        # 必须满足:
        # 1. 大趋势向上 (价格 > EMA200) +2
        # 2. RSI在回调区域 45-65 +2
        # 3. MACD金叉 +2
        # 4. 成交量萎缩(健康回调) +1
        
        conditions = pd.Series(False, index=dataframe.index)
        
        # 条件1: 大趋势向上
        trend_ok = dataframe['trend_up']
        
        # 条件2: RSI回调到合理区域
        # 关键修复: 趋势向上时 RSI 45-65 是健康回调，不是超买
        rsi_pullback = (dataframe['rsi'] > 45) & (dataframe['rsi'] < 65)
        
        # 条件3: MACD金叉
        macd_golden = dataframe['macd_cross_up']
        
        # 条件4: 健康回调(缩量或平量)
        healthy_pullback = dataframe['volume_ratio'] < 1.2
        
        # 综合评分
        score = (
            trend_ok.astype(int) * 2 +
            rsi_pullback.astype(int) * 2 +
            macd_golden.astype(int) * 2 +
            healthy_pullback.astype(int) * 1
        )
        
        # 入场: 分数>=5 且 趋势向上
        conditions = (score >= 5) & trend_ok
        
        # 额外信号: 底背驰 (高分加成)
        div_signal = dataframe['bullish_divergence'] & trend_ok
        
        dataframe['enter_long'] = (conditions | div_signal).astype(int)
        
        # ---------- 做空条件 (不做空，专注做多) ----------
        dataframe['enter_short'] = 0
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        # 顶背驰出场
        dataframe.loc[dataframe['bearish_divergence'], 'exit_long'] = 1
        
        # EMA死叉出场
        dataframe.loc[dataframe['ema_cross_down'], 'exit_long'] = 1
        
        return dataframe
    
    def custom_stoploss(self, trade: Trade, current_time: datetime,
                       current_rate: float, current_vol: float, **kwargs) -> float:
        """
        自定义止损 - 动态调整
        """
        return -0.05
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, 
                            current_rate: float, current_vol: float, 
                            remaining_space: float, max_rate: float,
                            current_profit: float, **kwargs) -> Optional[float]:
        """
        动态仓位调整
        """
        # 如果盈利超过5%，可以考虑加仓
        if current_profit > 0.05 and remaining_space > 0.02:
            return remaining_space * 0.3  # 最多加仓30%
        
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        交易确认
        """
        # 只交易SOL/USDT
        if pair != 'SOL/USDT':
            return False
        
        # 确认是做多
        if side != 'long':
            return False
        
        return True
    
    def bot_status(self) -> str:
        """
        状态报告
        """
        return "TitanV4 SOL合约策略运行中 - 4h周期"
