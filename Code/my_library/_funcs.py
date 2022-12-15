







#　取引通貨情報
def market_info():
    result = coincheck.load_markets()
    return result

# 板情報を取得する関数
def OrderBook():
    result = coincheck.fetchOrderBook(symbol=SYMBOL)
    return result

# 売り注文と買い注文を取得する関数
def ticker_info():
    result = coincheck.fetch_ticker(symbol=SYMBOL)
    return result

# 口座情報を取得する関数
def balance_info():
    result = coincheck.fetchBalance()
    return result

# 今出している注文をキャンセルする関数
# order_id は OpenOrder関数から取得可能
def OrderCancel(order_id):
    result = coincheck.cancel_order(
                    symbol = SYMBOL,  # 取引通貨
                    id = order_id,    # 注文ID
                    )
    return result

# 今出している注文を取得する関数
def OpenOrder():
    result = coincheck.fetchOpenOrders(symbol = SYMBOL)
    return result

# 買い注文を出す関数
def buy(price_value, btc_amount=0.005):
    order_symbol = 'BTC/JPY'   # 取引通貨
    type_value   = 'limit'     # 注文タイプ（limit:指値注文, market:成行注文）
    side_value   = 'buy'       # 買い(buy) or 売り(sell)

    # 注文コード
    order = coincheck.create_order(
                    symbol = order_symbol,  # 取引通貨
                    type = type_value,      # 注文タイプ
                    side= side_value,       # 買 or 売
                    amount= btc_amount,   # 取引数量
                    price=price_value,      # 指値価格
                    )
    return  order

# 売り注文を出す関数
def sell(price_value,btc_amount=0.005):
    order_symbol = 'BTC/JPY'   # 取引通貨
    type_value   = 'limit'     # 注文タイプ（limit:指値注文, market:成行注文）
    side_value   = 'sell'       # 買い(buy) or 売り(sell)

    # 注文コード
    order = coincheck.create_order(
                    symbol = order_symbol,  # 取引通貨
                    type = type_value,      # 注文タイプ
                    side= side_value,       # 買 or 売
                    amount= btc_amount,   # 取引数量
                    price=price_value,      # 指値価格
                    )
    return  order
    
# 買い注文, 売り注文が約定したかどうか把握する関数
# is_buy : True -> 買い注文約定
# is_sell : True -> 売り注文約定
def get_status(sell_amount):
    symbol = 'BTC/JPY'
    order_now = OpenOrder()
    b_info = balance_info()
    free = float(b_info['free']['BTC'])
    used = float(b_info['used']['BTC'])
    total = float(b_info['total']['BTC'])
    is_buy = False
    is_sell = False   

    # freeがBTC_AMOUT以下になる　つまり, 一部しか約定しない時があるならば, usedも0にならない時があるのでは?
    # freeが取引する数量の2倍以上はerrorの元なので, 通さない
    # Rate limit error : 売り注文の時しかでない？
    if len(order_now)==0 and used==0 and 0<free<BTC_AMOUNT*2:
        is_buy = True
        is_sell = True
        logger.info('here 1')
        
    # Rate limit error の時
    elif len(order_now)==0 and used==0 and free%BTC_AMOUNT==0 and free>=BTC_AMOUNT*2:
        is_buy=True
        is_sell = False
        logger.info('here 2')
        time.sleep(5)
        
    elif len(order_now)==0 and used==0 and free >= BTC_AMOUNT*2:
        is_buy=True
        is_sell = False
        logger.info('here 3')
        time.sleep(5)
    
    elif len(order_now)==0 and used>0:
        is_buy=True
        is_sell = False
        logger.info('here 4')
        time.sleep(5) 
        
    elif len(order_now)==1 :
        side = order_now[0]['side']
        if side == 'buy':
            is_sell = True
        elif side == 'sell':
            is_buy = True
        logger.info('here 5')
            
    elif len(order_now)==2 and used==sell_amount and free==0:
        is_buy = False
        is_sell = False 
    
    elif len(order_now)==2:
        side0 = order_now[0]['side']
        side1 = order_now[1]['side']
        
        if side0==side1=='buy' or side0==side1=='sell':
            # 古いほうの買い注文キャンセルしたい
            # 0 が古いほう
            if side0=='buy':logger.info('Two buy order')
            if side0=='sell':logger.info('Two sell order')
            
            old_order = order_now[0]['id']
            try:
                order_cancel = OrderCancel(old_order)
            except Exception as e:
                logger.info(e)
                logger.info('Cant cancel in get_status')
        
        else:
            logger.info('Buy and Sell order')
            logger.info('free {0}'.format(free))
            logger.info('used {0}'.format(used))
            
    elif len(order_now)==0 and used==0 and free==0:
        # free 持っている btc_amount
        # used 注文に出している btc_amount
        is_buy = False
        is_sell = True
        logger.info('here 6')
    
    else:
        logger.info('default')
        logger.info('free {0}'.format(free))
        logger.info('used {0}'.format(used))
        logger.info('order length :{0}'.format(len(order_now)))
        time.sleep(1)
        
    return is_buy, is_sell

# 板情報から最適な指値を計算する関数
def calc_best_price(size_thru=SIZE_THRU,buy_bias=BUY_BIAS,sell_bias=SELL_BIAS):
    orderbook = OrderBook()
    ask_list = orderbook['asks']
    bid_list = orderbook['bids']

    i = 0
    b_amount = 0
    while b_amount<=size_thru*BUY_BIAS:
        b_amount += bid_list[i][1]
        buy_price = bid_list[i][0]
        i+=1
        
    i=0
    a_amount = 0
    while a_amount<=size_thru*SELL_BIAS:
        a_amount += ask_list[i][1]
        sell_price = ask_list[i][0]
        i+=1
        
    ask = ask_list[0][0]
    bid = bid_list[0][0]
    
    return ask,bid,buy_price,sell_price

# ローソク足を取得する関数
def return_candles(min_='15m'):
    candles = coincheck.fetch_ohlcv(
        symbol=SYMBOL,     # 暗号資産[通貨]
        timeframe = min_,    # 時間足('1m', '5m', '1h', '1d')
        since=None,           # 取得開始時刻(Unix Timeミリ秒)
        limit=None,           # 取得件数(デフォルト:100、最大:500)
        params={}             # 各種パラメータ
        )
    return candles

# ローソク足からボラティリティを計算する関数
def calc_pct_rate(candles):
    open_ = candles[0][1]
    high_ = candles[0][2]
    low_ = candles[0][3]
    close_ = candles[0][4]
    volume_ = candles[0][5]
    
    pct_rate = (open_-close_)/close_
    return pct_rate, volume_

# LINE に通知する関数
def send_line_notify(message):
    """
    LINEに通知する
    """
    line_notify_token = TOKEN
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {message}'}
    requests.post(line_notify_api, headers = headers, data = data)
    
# 注文した値段と, 相場の価格の乖離率を計算する関数
def calc_deviation_rate(price):
    result = ticker_info()
    last_price = float(result['last'])
    dev_rate = np.abs((last_price - price) / price)
    return dev_rate

# 買いやすさ, 売りやすさを調整する関数
def return_trade_bias(pct_rate):
    trade_bias = 0

    if pct_rate>0:
        trade_bias = 100*np.tanh(pct_rate)
    elif pct_rate<0:
        trade_bias = 100*np.tanh(pct_rate)
    else:
        trade_bias = 0
        
    if np.abs(trade_bias)>=0.015:
        if trade_bias>0:
            trade_bias = 0.015
        else:
            trade_bias = -0.015
        
    return trade_bias

# spread を調整する関数
# 未実装
def return_spread(pct_rate,trade_freq,candles_hour):
    volume = candles_hour[0][5]
    spread = 0
    return spread

# 注文いれなおしたり, キャンセルする関数定義
def cancel_order(order_id):
    pass


# 指値更新の関数
# 段階損切なら, DP
# 様子見指値更新なら DELTA
def update_index_price(side,order_now,old_price,trade_amount,losscut_price):
    
    is_error = False
    is_rate_limit = False
    
    if side=='sell':            
        order_func = sell
    
    else:    # buy            
        order_func = buy
    
    price_tmp = old_price + losscut_price
    
    try:
        order_id = order_now[0]['id']
        order_cancel = OrderCancel(order_id)
        time.sleep(0.5)
        order_ = order_func(price_tmp,btc_amount=trade_amount)
        current_price = price_tmp
        return current_price, order_, is_error, is_rate_limit
        
    except Exception as e:
        is_error = True
        b_info = balance_info()
        free = float(b_info['free']['BTC'])
        
        # rate limit だった時   
        if free>=BTC_AMOUNT*2 and side=='sell':
            logger.info('Rate limit')
            is_rate_limit = True
            time.sleep(5)
        else:
            logger.info('in Update '+side)
            logger.info(e)
        
        return price_tmp, None, is_error, is_rate_limit
    
