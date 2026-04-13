# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
高胜率短线策略 - ShortTermStrategy
结合 RSI + 布林带 + MACD + EMA 均线交叉 + 成交量确认
目标胜率: 80%+
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
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class ShortTermStrategy(IStrategy):
    """
    高胜率短线策略
    使用多指标共振来提高入场信号的准确性
    
    入场条件:
    1. RSI 超卖 (< 30) + 价格触及布林下轨
    2. MACD 金叉 + EMA 均线多头排列
    3. 成交量放大确认
    
    出场条件:
    1. RSI 超买 (> 70)
    2. 价格触及布林上轨
    3. MACD 死叉
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # 是否支持做空
    can_short: bool = False

    # ========== 基础参数 ==========
    timeframe = "4h"
    
    # 启动所需的蜡烛图数量（布林带需要200个）
    startup_candle_count = 200
    
    # 只处理新蜡烛
    process_only_new_candles = True
    
    # 是否使用退出信号
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ========== 止损止盈 ==========
    stoploss = -0.03  # -3% 止损
    
    # 追踪止损
    trailing_stop = True
    trailing_stop_positive = 0.025  # 盈利2.5%后启动追踪
    trailing_stop_positive_offset = 0.03  # 追踪止损距离3%
    trailing_only_offset_is_reached = True
    
    # 最小 ROI（按时间递减）
    minimal_roi = {
        "0": 0.05,      # 0分钟: 5% 宽止盈!
        "240": 0.03,    # 4小时: 3%
        "720": 0.02,    # 12小时: 2%
    }

    # ========== 订单类型 ==========
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # ========== Hyperopt 可调参数 ==========
    
    # RSI 参数
    buy_rsi = IntParameter(low=25, high=40, default=35, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=65, high=80, default=70, space="sell", optimize=True, load=True)
    
    # MACD 参数
    buy_macd_fast = IntParameter(low=8, high=16, default=12, space="buy", optimize=True, load=True)
    buy_macd_signal = IntParameter(low=20, high=32, default=26, space="buy", optimize=True, load=True)
    sell_macd_fast = IntParameter(low=8, high=16, default=12, space="sell", optimize=True, load=True)
    sell_macd_signal = IntParameter(low=20, high=32, default=26, space="sell", optimize=True, load=True)
    
    # 布林带参数
    buy_bb_percent = DecimalParameter(low=0.1, high=0.25, default=0.2, space="buy", optimize=True, load=True)
    sell_bb_percent = DecimalParameter(low=0.8, high=1.0, default=1.0, space="sell", optimize=True, load=True)
    
    # EMA 均线参数
    buy_ema_fast = IntParameter(low=5, high=15, default=9, space="buy", optimize=True, load=True)
    buy_ema_slow = IntParameter(low=20, high=50, default=21, space="buy", optimize=True, load=True)
    sell_ema_fast = IntParameter(low=5, high=15, default=9, space="sell", optimize=True, load=True)
    sell_ema_slow = IntParameter(low=20, high=50, default=21, space="sell", optimize=True, load=True)
    
    # 成交量倍数
    buy_volume_min = DecimalParameter(low=1.0, high=2.0, default=1.2, space="buy", optimize=True, load=True)
    
    # 多指标确认开关
    confirm_trend翻转 = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    confirm_macd_cross = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    confirm_ema_cross = BooleanParameter(default=True, space="buy", optimize=True, load=True)

    def informative_pairs(self):
        """
        定义额外的 informative pairs
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        添加技术指标
        """
        # ========== RSI ==========
        dataframe["rsi"] = ta.RSI(dataframe)
        
        # ========== MACD ==========
        macd = ta.MACD(dataframe, fastperiod=int(self.buy_macd_fast.value), signalperiod=int(self.buy_macd_signal.value))
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        
        # 卖出 MACD
        macd_sell = ta.MACD(dataframe, fastperiod=int(self.sell_macd_fast.value), signalperiod=int(self.sell_macd_signal.value))
        dataframe["macd_sell"] = macd_sell["macd"]
        dataframe["macdsignal_sell"] = macd_sell["macdsignal"]
        dataframe["macdhist_sell"] = macd_sell["macdhist"]

        # ========== 布林带 ==========
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), 
            window=20, 
            stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) /
            dataframe["bb_middleband"]
        )

        # ========== EMA 均线 ==========
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_fast.value))
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_slow.value))
        
        # 用于卖出的 EMA
        dataframe["ema_fast_sell"] = ta.EMA(dataframe, timeperiod=int(self.sell_ema_fast.value))
        dataframe["ema_slow_sell"] = ta.EMA(dataframe, timeperiod=int(self.sell_ema_slow.value))

        # ========== 均线交叉信号 ==========
        dataframe["ema_cross_above"] = qtpylib.crossed_above(
            dataframe["ema_fast"], 
            dataframe["ema_slow"]
        )
        dataframe["ema_cross_below"] = qtpylib.crossed_below(
            dataframe["ema_fast"], 
            dataframe["ema_slow"]
        )
        
        # 卖出均线交叉
        dataframe["ema_cross_above_sell"] = qtpylib.crossed_above(
            dataframe["ema_fast_sell"], 
            dataframe["ema_slow_sell"]
        )
        dataframe["ema_cross_below_sell"] = qtpylib.crossed_below(
            dataframe["ema_fast_sell"], 
            dataframe["ema_slow_sell"]
        )

        # ========== MACD 交叉信号 ==========
        dataframe["macd_cross_above"] = qtpylib.crossed_above(
            dataframe["macd"], 
            dataframe["macdsignal"]
        )
        dataframe["macd_cross_below"] = qtpylib.crossed_below(
            dataframe["macd"], 
            dataframe["macdsignal"]
        )
        
        # 卖出 MACD 交叉
        dataframe["macd_cross_above_sell"] = qtpylib.crossed_above(
            dataframe["macd_sell"], 
            dataframe["macdsignal_sell"]
        )
        dataframe["macd_cross_below_sell"] = qtpylib.crossed_below(
            dataframe["macd_sell"], 
            dataframe["macdsignal_sell"]
        )

        # ========== 成交量 ==========
        dataframe["volume_avg"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_avg"]
        
        # ========== ATR (用于止损) ==========
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        
        # ========== 趋势判断 (ADX) ==========
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        
        # ==========  Stochastic RSI ==========
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe["stoch_rsi_k"] = stoch_rsi["fastk"]
        dataframe["stoch_rsi_d"] = stoch_rsi["fastd"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义入场信号
        """
        dataframe["enter_long"] = 0
        
        # 条件1: RSI 超卖 + 布林下轨
        cond1_rsi_bb = (
            (dataframe["rsi"] < self.buy_rsi.value) &
            (dataframe["close"] <= dataframe["bb_lowerband"] * 1.01) &
            (dataframe["volume"] > 0)
        )
        
        # 条件2: MACD 金叉
        cond2_macd = (
            qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"]) &
            (dataframe["macd"] < 0)  # MACD 在零轴下方
        )
        
        # 条件3: EMA 多头排列 (快线在慢线上方)
        cond3_ema = (
            dataframe["ema_fast"] > dataframe["ema_slow"] &
            dataframe["ema_cross_above"]
        )
        
        # 条件4: 成交量放大
        cond4_volume = dataframe["volume_ratio"] >= float(self.buy_volume_min.value)
        
        # 条件5: ADX 确认趋势存在
        cond5_adx = dataframe["adx"] > 20
        
        # 综合入场信号 (多指标共振)
        buy_signal = cond1_rsi_bb.copy()
        
        if self.confirm_macd_cross.value:
            buy_signal = buy_signal & cond2_macd
            
        if self.confirm_ema_cross.value:
            buy_signal = buy_signal & cond3_ema
            
        buy_signal = buy_signal & cond4_volume & cond5_adx
        
        dataframe.loc[buy_signal, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        定义出场信号
        """
        dataframe["exit_long"] = 0
        
        # 条件1: RSI 超买
        cond1_rsi = (
            (dataframe["rsi"] > self.sell_rsi.value) &
            (dataframe["volume"] > 0)
        )
        
        # 条件2: 价格触及布林上轨
        cond2_bb = (
            (dataframe["close"] >= dataframe["bb_upperband"] * 0.99)
        )
        
        # 条件3: MACD 死叉
        cond3_macd = qtpylib.crossed_below(dataframe["macd_sell"], dataframe["macdsignal_sell"])
        
        # 条件4: EMA 空头排列
        cond4_ema = (
            dataframe["ema_fast_sell"] < dataframe["ema_slow_sell"]
        )
        
        # 综合出场信号
        exit_signal = cond1_rsi | cond2_bb | cond3_macd
        
        if self.confirm_ema_cross.value:
            exit_signal = exit_signal & cond4_ema
            
        dataframe.loc[exit_signal, "exit_long"] = 1

        return dataframe

    # ========== 自定义止损 ==========
    def custom_stoploss(
        self, 
        pair: str, 
        trade: Trade, 
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs
    ) -> Optional[float]:
        """
        自定义止损逻辑
        当利润低于 -3% 时加强止损
        """
        if current_profit < -0.03:
            return -0.03  # 锁定最大亏损3%
        return self.stoploss

    # ========== 自定义ROI ==========
    def custom_roi(
        self, 
        pair: str, 
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[float]:
        """
        自定义 ROI - 根据持仓时间动态调整
        """
        # 如果已经有不错的利润，可以早点出场
        if current_profit > 0.04:
            return 0.015  # 4%以上利润，1.5%就出场
        elif current_profit > 0.025:
            return 0.01
        elif current_profit > 0.015:
            return 0.008
        return None  # 使用默认 ROI

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "blue"},
            "ema_slow": {"color": "red"},
            "bb_upperband": {"color": "gray"},
            "bb_middleband": {"color": "gray"},
            "bb_lowerband": {"color": "gray"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "red"},
            },
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
                "macdhist": {"color": "purple", "type": "bar"},
            },
            "BB": {
                "bb_percent": {"color": "green"},
            },
            "Volume": {
                "volume_ratio": {"color": "teal"},
            },
        },
    }
