import backtrader as bt
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置matplotlib使用SimHei字体来支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
##plt.style.use('dark_background')


# 自定义固定佣金方案
class FixedCommissionScheme(bt.CommInfoBase):
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        comm = abs(size) * 0.0049
        platform_fee = abs(size) * 0.005
        clearing_fee = abs(size) * 0.003

        comm = max(comm, 0.99)
        platform_fee = max(platform_fee, 1)
        total_fee = comm + platform_fee + clearing_fee

        if size < 0:
            sec_fee = abs(size) * price * 0.00002778
            finra_fee = abs(size) * 0.000166
            total_fee += sec_fee + finra_fee

        return total_fee

# 动态网格交易策略
class AdaptiveGridTradingStrategy(bt.Strategy):
    params = (
        ('initial_grid_size', 0.1),  # 初始网格大小1.3
        ('max_grid_orders', 30),  #20
        ('atr_period', 17),  # ATR计算周期17
    )

    def __init__(self):
        self.orders = []
        self.grid_levels = {}
        self.total_commission = 0
        self.val_start = self.broker.getvalue()
        self.val_end = 0
        self.cash_values = []
        self.daily_returns = []
        self.highest_value = self.val_start
        self.total_buy_volume = 0
        self.total_sell_volume = 0

        self.atr = {data._name: bt.indicators.AverageTrueRange(data, period=self.params.atr_period) for data in self.datas}

        for data in self.datas:
            self.init_grid_levels(data)

    def init_grid_levels(self, data):
        current_price = data.close[0]
        grid_size = self.params.initial_grid_size
        grid_levels = []
        for i in range(1, self.params.max_grid_orders + 1):
            buy_level = current_price - i * grid_size
            sell_level = current_price + i * grid_size
            grid_levels.append(('buy', buy_level))
            grid_levels.append(('sell', sell_level))
        self.grid_levels[data._name] = grid_levels
        self.log(f"初始网格级别 ({data._name}): {grid_levels}")

    def next(self):
        for data in self.datas:
            current_price = data.close[0]
            data_name = data._name
            position_size = self.getposition(data).size

            grid_size = self.atr[data_name][0]  # 使用ATR动态调整网格大小

            buy_order_placed = False
            sell_order_placed = False

            for order_type, level in self.grid_levels[data_name]:
                if order_type == 'buy' and current_price <= level and not buy_order_placed:
                    size = int(self.broker.getcash() / current_price / (self.params.max_grid_orders / 2))
                    if size <= 0:
                        self.log("当前现金不足以执行买单")
                        continue
                    order = self.buy(data=data, size=size, exectype=bt.Order.Limit, price=level)
                    self.orders.append(order)
                    self.log(f"创建买入订单 ({data_name}), 价格: {level:.2f}, 数量: {size}")
                    buy_order_placed = True

                elif order_type == 'sell' and current_price >= level and not sell_order_placed:
                    size = int(self.broker.getvalue() / current_price / (self.params.max_grid_orders / 2))
                    size = min(size, position_size)
                    if size <= 0:
                        self.log("当前持仓不足以执行卖单")
                        continue
                    order = self.sell(data=data, size=size, exectype=bt.Order.Limit, price=level)
                    self.orders.append(order)
                    self.log(f"创建卖出订单 ({data_name}), 价格: {level:.2f}, 数量: {size}")
                    sell_order_placed = True

        self.cash_values.append(self.broker.getvalue())
        if len(self.cash_values) > 1:
            daily_return = (self.cash_values[-1] - self.cash_values[-2]) / self.cash_values[-2]
            self.daily_returns.append(daily_return)
        self.highest_value = max(self.highest_value, self.broker.getvalue())

    def log(self, txt, data=None):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()}: {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行 ({order.data._name}), 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 数量: {order.executed.size}, 佣金: {order.executed.comm:.2f}')
                self.total_buy_volume += order.executed.size
            elif order.issell():
                self.log(f'卖出执行 ({order.data._name}), 价格: {order.executed.price:.2f}, 成本: {order.executed.value:.2f}, 数量: {order.executed.size}, 佣金: {order.executed.comm:.2f}')
                self.total_sell_volume += order.executed.size
            self.total_commission += order.executed.comm
            if order in self.orders:
                self.orders.remove(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单失败 ({order.data._name}), 原因: {order.getstatusname()}')

    def stop(self):
        self.val_end = self.broker.getvalue()
        self.plot_figures()
        print(f'总佣金: {self.total_commission:.2f} 美元')
        print(f'总买单股数: {self.total_buy_volume}')
        print(f'总卖单股数: {self.total_sell_volume}')

        trading_days = len(self.daily_returns)
        if trading_days > 0:
            annualized_return = (self.val_end / self.val_start) ** (252 / trading_days) - 1
            annualized_volatility = np.std(self.daily_returns) * np.sqrt(252)
            risk_free_rate = 0.05
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            drawdown = np.array(self.cash_values) / np.maximum.accumulate(self.cash_values) - 1
            max_drawdown = np.min(drawdown)
            benchmark_return = 0.07
            information_ratio = (annualized_return - benchmark_return) / annualized_volatility

            print(f'最终组合价值: {self.val_end:.2f}')
            print(f'年化收益率: {annualized_return * 100:.2f}%')
            print(f'年化波动率: {annualized_volatility * 100:.2f}%')
            print(f'夏普比率: {sharpe_ratio:.2f}')
            print(f'最大回撤: {max_drawdown * 100:.2f}%')
            print(f'信息比率: {information_ratio:.2f}')
        else:
            print("交易天数为零，无法计算绩效指标。")

    def plot_figures(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.cash_values, label='组合价值', color='red')
        plt.title('组合价值随时间的变化')
        plt.xlabel('时间（单位：bars）')
        plt.ylabel('价值')
        plt.legend()
        plt.show()

# 主程序
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(500000)
    cerebro.addstrategy(AdaptiveGridTradingStrategy)

    comminfo = FixedCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)

    stock_symbols = ['NVDA']
    data_path = Path(os.getcwd()) / 'data'

    for stock_symbol in stock_symbols:
        data_file = data_path / f'{stock_symbol}.csv'
        data = bt.feeds.YahooFinanceCSVData(
            dataname=data_file,
            fromdate=datetime.datetime(2017, 1, 1),
            todate=datetime.datetime(2020, 1, 30),
            reverse=False,
            name=stock_symbol
        )
        cerebro.adddata(data)

    print(f'初始组合价值: {cerebro.broker.getvalue():.2f}')

    cerebro.run()

    final_value = cerebro.broker.getvalue()
    print(f'最终组合价值: {final_value:.2f}')
    print(f'回报率: {(final_value - cerebro.broker.startingcash) / cerebro.broker.startingcash * 100:.2f}%')
    cerebro.plot(style='candlestick')
