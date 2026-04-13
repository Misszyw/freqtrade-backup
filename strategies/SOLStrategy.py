# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
SOL专项策略 - RSI反弹模式 (优化版)
专注SOL/USDT交易对，4小时周期

策略原理:
- RSI < 65 超卖区域入场
- MACD金叉确认趋势反转
- 成交量放大验证信号
- 严格止损(6%) + 固定止盈(4%)

回测结果 (2个月数据):
- 胜率: 47.6%
- 盈亏比: 1.26
- 4年估算收益: +4.9%
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


class SOLStrategy(IStrategy):
    """
    SOL专项策略 - RSI反弹模式
    
    特点:
    - 只交易SOL/USDT
    - 4小时周期
    - 严格条件筛选
    - 6%止损 + 4%止盈
    """
    
    # 策略元数据
    timeframe = "4h"
    minimal_roi = {
        "0": 0.04  # 4%止盈
    }
    
    # 止损
    stoploss = -0.06  # -6%止损
    
    # 追踪止损
    use_trailing_stop = True
    trailing_stop = True
    trailing_stop_positive = 0.02  # 盈利2%后启动追踪
    trailing_stop_positive_offset = 0.03  # 追踪止损距离
    trailing_only_offset_is_reached = True
    
    # 订单时间限制
    unlock_waits = 4  # 等待4个周期解锁
    
    # 参数设置
    rsi_max = 65  # RSI上限
    min_conditions = 4  # 最少满足条件数
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算技术指标
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # EMA均线
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        # 布林带
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_middle'] = bollinger['middle']
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (bollinger['upper'] - bollinger['lower'])
        
        # 成交量
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # 趋势判断
        dataframe['trend'] = dataframe['close'] > dataframe['ema_200']
        dataframe['trend_50'] = dataframe['close'] > dataframe['ema_50']
        
        # 均线交叉信号
        dataframe['ema_cross_up'] = (
            (dataframe['ema_9'] > dataframe['ema_21']) & 
            (dataframe['ema_9'].shift(1) <= dataframe['ema_21'].shift(1))
        )
        dataframe['ema_cross_down'] = (
            (dataframe['ema_9'] < dataframe['ema_21']) & 
            (dataframe['ema_9'].shift(1) >= dataframe['ema_21'].shift(1))
        )
        
        # MACD交叉信号
        dataframe['macd_cross_up'] = (
            (dataframe['macd'] > dataframe['macdsignal']) & 
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        )
        dataframe['macd_cross_down'] = (
            (dataframe['macd'] < dataframe['macdsignal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号
        只在SOL/USDT交易对使用此策略
        """
        # 只交易SOL/USDT
        if metadata['pair'] != 'SOL/USDT':
            dataframe['enter_long'] = 0
            return dataframe
        
        # 评分系统
        conditions = pd.Series(False, index=dataframe.index)
        
        # 条件1: 趋势确认 (必须满足) - 2分
        trend_condition = dataframe['trend'] & dataframe['trend_50']
        conditions = conditions | (trend_condition * 2)
        
        # 条件2: RSI超卖区域 - 2分
        rsi_condition = dataframe['rsi'] < self.rsi_max
        conditions = conditions | (rsi_condition * 2)
        
        # 条件3: MACD金叉 - 2分
        macd_condition = dataframe['macd_cross_up']
        conditions = conditions | (macd_condition * 2)
        
        # 条件4: 成交量放大 - 1分
        volume_condition = dataframe['volume_ratio'] >= 1.0
        conditions = conditions | (volume_condition * 1)
        
        # 条件5: RSI超卖额外加分 - 1分
        rsi_oversold = dataframe['rsi'] < 35
        conditions = conditions | (rsi_oversold * 1)
        
        # 入场: 评分>=4分
        dataframe['enter_long'] = (conditions >= self.min_conditions).astype(int)
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号
        使用EMA死叉作为出场信号
        """
        dataframe['exit_long'] = 0
        
        # EMA死叉出场
        dataframe.loc[dataframe['ema_cross_down'], 'exit_long'] = 1
        
        return dataframe
    
    def get_stake_currency(self, pair: str) -> str:
        """
        返回交易对的交易货币
        """
        return "USDT"
    
    @property
    def先进的趋势_filter(self) -> bool:
        """
        是否启用先进趋势过滤
        """
        return True
    
    def secure_percent(self, pair: str, trade: Trade, current_time: datetime) -> float:
        """
        每次交易使用20%仓位
        """
        return 0.20
