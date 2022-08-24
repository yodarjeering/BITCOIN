import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


def standarize(df):
    # ddof = 0 : 分散
    # ddof = 1 : 不偏分散
    df = (df - df.mean())/df.std(ddof=0)
    return df

class DataFramePreProcessing():

    
    def __init__(self, path_, is_daw=False):
        self.path_ = path_
        self.is_daw = is_daw

        
    def load_df(self):
        if self.is_daw:
            d='d'
        else:
            d=''
        FILE = glob.glob(self.path_)
        df = pd.read_csv(FILE[0])
        df = df.rename(columns={df.columns[0]:'nan',df.columns[1]:'nan',df.columns[2]:'nan',\
                                    df.columns[3]:'day',df.columns[4]:'nan',df.columns[5]:d+'open',\
                                    df.columns[6]:d+'high',df.columns[7]:d+'low',df.columns[8]:d+'close',\
                df.columns[9]:d+'volume',})
        df = df.drop('nan',axis=1)
        df = df.drop(df.index[0])
        df['day'] = pd.to_datetime(df['day'],format='%Y/%m/%d')
        df.set_index('day',inplace=True)

        return df.astype(float)
    
class PlotTrade():
    
    
    def __init__(self, df_chart,label=''):
        self.df_chart = df_chart
        plt.clf()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.plot(self.df_chart,label=label)
        plt.legend()
        
    def add_span(self, start_time,end_time):
        self.ax.axvspan(start_time, end_time, color="gray", alpha=0.3)
        
    
    def add_plot(self, df_plot,label=''):
        self.ax.plot(df_plot,label=label)
        plt.legend()
        
        
    def show(self):
        self.ax.grid()
        labels = self.ax.get_xticklabels()
        plt.setp(labels, rotation=15, fontsize=12)
        plt.show()

class ValidatePlot(PlotTrade):


    
    
    def __init__(self, df_chart, is_validate=False):
        pass
        
    def add_span(self, start_time,end_time):
        pass
        
    
    def add_plot(self, df_plot):
        pass
        
        
    def show(self):
        pass
    
class MakeTrainData():
    

    def __init__(self, df_con, test_rate=0.9, is_bit_search=False,is_category=True,ma_short=5,ma_long=25):
        self.df_con = df_con
        self.test_rate = test_rate
        self.is_bit_search = is_bit_search
        self.is_category = is_category
        self.ma_short = ma_short
        self.ma_long = ma_long

                
    def add_ma(self):
        df_process = self.df_con.copy()
        df_process['ma_short'] = df_process['close'].rolling(self.ma_short).mean()
        df_process['ma_long']  = df_process['close'].rolling(self.ma_long).mean()
        df_process['std_short'] = df_process['close'].rolling(self.ma_short).std()
        df_process['std_long']  = df_process['close'].rolling(self.ma_long).std()
        df_process['ema_short'] = df_process['close'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['ema_long'] = df_process['close'].ewm(span=self.ma_long, adjust=False).mean()
        df_process['macd'] = df_process['ema_short'] - df_process['ema_long']
        df_process['macd_signal_short'] = df_process['macd'].ewm(span=self.ma_short, adjust=False).mean()
        df_process['macd_signal_long'] = df_process['macd'].ewm(span=self.ma_long, adjust=False).mean()
        return df_process
                
   
        
    def make_data(self,is_check=False):
        x = pd.DataFrame(index=self.df_con.index)
        # この書き方は環境によってはエラー
        # x.index = self.df_con.index
        df_con = self.df_con.copy()
        df_ma = self.add_ma()
        end_point = -1
        if is_check:
            end_point = len(self.df_con)
        else:
            end_point = len(self.df_con)-1
        
        # ダウ変化率
        dawp_5 = df_con['dclose'].iloc[:-5]
        dawp_5.index = df_con.index[5:]
        x['dawp_5'] = dawp_5
        dawp_0 = df_con['dclose']
        x['dawp_0'] = dawp_0
        
        # 日経変化率
        nikkeip_5 = df_con['pclose'].iloc[:-5]
        nikkeip_5.index = df_con.index[5:]
        x['nikkeip_5'] = nikkeip_5
        nikkeip_0 = df_con['pclose']
        x['nikkeip_0'] = nikkeip_0
        
        # high - low 変化率
        high_low = (df_con['high']-df_con['low'])/df_con['close']
        x['diff_rate'] = high_low
        
        # close - open 変化率
        close_open = (df_con['close']-df_con['open'])/df_con['close']
        x['close_open'] = close_open
        
        # 売買量変化率
        nikkei_volumep = df_con['volume'].pct_change()
        x['nikkei_volumep'] = nikkei_volumep
        
        # 短期標準偏差ベクトル
        std_s_5 = df_ma['std_short'].iloc[:-5]
        std_s_5.index = df_ma.index[5:]
        x['std_s_5'] = std_s_5
        std_s_0 = df_ma['std_short']
        x['std_s_0'] = std_s_0
        
        # 長期標準偏差ベクトル
        std_l_5 = df_ma['std_long'].iloc[:-5]
        std_l_5.index = df_ma.index[5:]
        x['std_l_5'] = std_l_5
        std_l_0 = df_ma['std_long']
        x['std_l_0'] = std_l_0
        
        # 短期移動平均ベクトル
        vec_s_5 = (df_ma['ma_short'].diff(5)/5)
        x['vec_s_5'] = vec_s_5
        vec_s_1 = (df_ma['ma_short'].diff(1)/1)
        x['vec_s_1'] = vec_s_1
        
        # 長期移動平均ベクトル
        vec_l_5 = (df_ma['ma_long'].diff(5)/5)
        x['vec_l_5'] = vec_l_5
        vec_l_1 = (df_ma['ma_long'].diff(1)/1)
        x['vec_l_1'] = vec_l_1
        
#         移動平均乖離率
        x['d_MASL'] = df_ma['ma_short']/df_ma['ma_long']

#          ema短期のベクトル
        emavec_s_5 = (df_ma['ema_short'].diff(5)/5)
        x['emavec_s_5'] = emavec_s_5
        emavec_s_1 = (df_ma['ema_short'].diff(1)/1)
        emavec_s_1.index = df_ma.index
        x['emavec_s_1'] = emavec_s_1
    
        # ema長期ベクトル
        emavec_l_5 = (df_ma['ema_long'].diff(5)/5)
        x['emavec_l_5'] = emavec_l_5
        emavec_l_1 = (df_ma['ema_long'].diff(1)/1)
        x['emavec_l_1'] = emavec_l_1

        #         EMA移動平均乖離率
        x['d_EMASL'] = df_ma['ema_short']/df_ma['ema_long']
        
        # macd
        macd = df_ma['macd']
        x['macd'] = macd
        macd_signal_short = df_ma['macd_signal_short']
        x['macd_signal_short'] = macd_signal_short
        macd_signal_long = df_ma['macd_signal_long']
        x['macd_signal_long'] = macd_signal_long
            
        # 短期相関係数
        df_tmp1 = df_con[['close','daw_close']].rolling(self.ma_short).corr()
        corr_short = df_tmp1.drop(df_tmp1.index[0:-1:2])['close']
        corr_short = corr_short.reset_index().set_index('day')['close']
        x['corr_short'] = corr_short
        
        # 長期相関係数
        df_tmp2 = df_con[['close','daw_close']].rolling(self.ma_long).corr()
        corr_long = df_tmp2.drop(df_tmp2.index[0:-1:2])['close']
        corr_long = corr_long.reset_index().set_index('day')['close']
        x['corr_long'] = corr_long
        
        # 歪度
        skew_short = df_con['close'].rolling(self.ma_short).skew()
        x['skew_short'] = skew_short
        skew_long = df_con['close'].rolling(self.ma_long).skew()
        x['skew_long'] = skew_long
        
        # 尖度
        kurt_short = df_con['close'].rolling(self.ma_short).kurt()
        x['kurt_short'] = kurt_short
        kurt_long = df_con['close'].rolling(self.ma_long).kurt()
        x['kurt_long'] = kurt_long
        
        # RSI 相対力指数
        df_up = df_con['dclose'].copy()
        df_down = df_con['dclose'].copy()
        df_up[df_up<0] = 0
        df_down[df_down>0] = 0
        df_down *= -1
        sims_up = df_up.rolling(self.ma_short).mean()
        sims_down = df_down.rolling(self.ma_short).mean()
        siml_up = df_up.rolling(self.ma_long).mean()
        siml_down = df_down.rolling(self.ma_long).mean()
        RSI_short = sims_up / (sims_up + sims_down) * 100
        RSI_long = siml_up / (siml_up + siml_down) * 100
        x['RSI_short'] = RSI_short
        x['RSI_long'] = RSI_long
        
        
        open_ =  df_con['open']
        high_ = df_con['high']
        low_ = df_con['low']
        close_ = df_con['close']

#        Open Close 乖離率
        x['d_OC'] = open_/close_

#       High low 乖離率
        x['d_HL'] = high_/low_
        df_atr = pd.DataFrame(index=high_.index)
        df_atr['high_low'] = high_ - low_
        df_atr['high_close'] = high_ - close_
        df_atr['close_low_abs'] =  (close_ - low_).abs()
        tr = pd.DataFrame(index=open_.index)
        tr['TR'] = df_atr.max(axis=1)

        # ATR
        x['ATR_short'] = tr['TR'].rolling(self.ma_short).mean()
        x['ATR_long'] =  tr['TR'].rolling(self.ma_long).mean()
        x['d_ATR'] = x['ATR_short']/x['ATR_long']
        x['ATR_vecs5'] = (x['ATR_short'].diff(5)/1)
        x['ATR_vecs1'] = (x['ATR_short'].diff(1)/1)
        x['ATR_vecl5'] = (x['ATR_long'].diff(5)/1)
        x['ATR_vecl1'] = (x['ATR_long'].diff(1)/1)
        
        today_close = df_con['close']
        yesterday_close = df_con['close'].iloc[:-1]
        yesterday_close.index = df_con.index[1:]
#        騰落率
#       一度も使用されていなかったため, 削除
        # x['RAF'] =  (today_close/yesterday_close -1)
        x = x.iloc[self.ma_long:end_point]
        x_check = x
#        この '4' は　std_l5 など, インデックスをずらす特徴量が, nanになってしまう分の日数を除くためのもの
        # yについても同様
        x_train = x.iloc[self.ma_short-1:int(len(x)*self.test_rate)]
        x_test  = x.iloc[int(len(x)*self.test_rate):]


        if not is_check:
            
            y_train,y_test = self.make_y_data(x,self.df_con,end_point)
            return x_train, y_train, x_test, y_test
        
        
        else:
            x_check = x_check.iloc[self.ma_short-1:]
            chart_ = self.df_con.loc[x_check.index]
            
            return x_check,chart_


    def make_y_data(self,x,df_con,end_point):
        y = []
        for i in range(self.ma_long,end_point):
            tommorow_close = df_con['close'].iloc[i+1]
            today_close    = df_con['close'].iloc[i]
            if tommorow_close>today_close:
                y.append(1)
            else:
                y.append(0)
            
        y_train = y[self.ma_short-1:int(len(x)*self.test_rate)]
        y_test  = y[int(len(x)*self.test_rate):]
        return y_train,y_test


class LearnXGB():
    
    
    def __init__(self, num_class=2):
        self.model = xgb.XGBClassifier()
        self.x_test = None
        self.num_class = num_class
        if num_class==2:
            self.MK = MakeTrainData
        elif num_class==3:
            self.MK = MakeTrainData3
    

    def learn_xgb(self, path_tpx, path_daw, test_rate=0.8, param_dist='None'):
        x_train,y_train,x_test,y_test = self.make_xgb_data(path_tpx,path_daw,test_rate)
        
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
            'n_estimators':16,
            'max_depth':4,
            'random_state':0
            }

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))

        print(classification_report(np.array(y_test), hr_pred))
        _, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model


    def learn_xgb2(self,x_train,y_train,x_test,y_test,param_dist='None'):
        if param_dist=='None':
#             Grid search で求めたパラメタ 2021/11/21
            param_dist = { 
                'n_estimators':16,
                'max_depth':4,
                'random_state':0
                }

        xgb_model = xgb.XGBClassifier(**param_dist)
        hr_pred = xgb_model.fit(x_train.astype(float), np.array(y_train), eval_metric='logloss').predict(x_test.astype(float))
        print("---------------------")
        y_proba_train = xgb_model.predict_proba(x_train)[:,1]
        y_proba = xgb_model.predict_proba(x_test)[:,1]

        if self.num_class==2:
            print('AUC train:',roc_auc_score(y_train,y_proba_train))    
            print('AUC test :',roc_auc_score(y_test,y_proba))

        print(classification_report(np.array(y_test), hr_pred))
        _, ax = plt.subplots(figsize=(12, 10))
        xgb.plot_importance(xgb_model,ax=ax) 
        self.model = xgb_model
        

    def make_state(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))
        chart_ = df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        return state_, chart_
        
        
    def make_xgb_data(self, path_tpx, path_daw, test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=test_rate)
        x_train, y_train, x_test, y_test = mk.make_data()
        return x_train,y_train,x_test,y_test
    
    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose']],axis = 1,join='inner').astype(float)
        df_con['pclose'] = df_con['close'].pct_change()
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con
    
    
    def for_ql_data(self, path_tpx, path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)

        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        state_ = self.model.predict_proba(x_check.astype(float))

        chart_ = mk.df_con['close'].loc[x_check.index[0]:x_check.index[-1]]
        state_ = pd.DataFrame(state_)
        state_['day'] = chart_.index
        
        state_.reset_index(inplace=True)
        state_.set_index('day',inplace=True)
        state_.drop('index',axis=1,inplace=True)
        return state_, chart_
    
    
    def predict_tomorrow(self, path_tpx, path_daw, alpha=0.5, strategy='normal', is_online=False, is_valiable_strategy=False,start_year=2021,start_month=1,end_month=12,is_observed=False,is_validate=False):
        xl = XGBSimulation(self,alpha=alpha)
        xl.simulate(path_tpx,path_daw,is_validate=is_validate,strategy=strategy,is_variable_strategy=is_valiable_strategy,start_year=start_year,start_month=start_month,end_month=end_month,is_observed=is_observed,is_online=is_online)
        self.xl = xl
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con)
        x_check, chart_ = mk.make_data(is_check=True)
        tomorrow_predict = self.model.predict_proba(x_check)
        label = self.get_tomorrow_label(tomorrow_predict,strategy, is_valiable_strategy)
        print("is_bought",xl.is_bought)
        print("df_con in predict_tomorrow",df_con.index[-1])
        print("today :",x_check.index[-1])
        print("tomorrow UP possibility", tomorrow_predict[-1,1])
        print("label :",label)


    def get_tomorrow_label(self, tomorrow_predict,strategy, is_valiable_strategy):
        label = "STAY"
        df_con = self.xl.df_con
        if is_valiable_strategy:
            i = len(df_con)-2
            strategy = self.xl.return_grad(df_con, index=i,gamma=0, delta=0)
        
        if strategy == 'normal':
            if tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label =  "SELL"
            else:
                label = "STAY"
        
        elif strategy == 'reverse':
            if 1-tomorrow_predict[-1,1] > self.xl.alpha:
                label = "BUY"
            elif tomorrow_predict[-1,1] > self.xl.alpha:
                label = "SELL"
            else:
                label = "STAY"

        return label


class LearnClustering(LearnXGB):


    def __init__(self,n_cluster=8,random_state=0,width=40,stride=5,strategy_table=None):
        super(LearnClustering,self).__init__()
        self.model : KMeans = None
        self.n_cluster = n_cluster
        self.width = width
        self.stride = stride
        self.n_label = None
        self.wave_dict = None
        self.strategy_table = strategy_table
        self.random_state=random_state



    def make_x_data(self,close_,width=20,stride=5,test_rate=0.8):
        length = int(len(close_)*test_rate)
        close_ = close_.iloc[:length]
        # close_ = standarize(close_)
        # close_list = close_.tolist()

        x = []
        z = []
        for i in range(0,length-width,stride):
            x.append(standarize(close_.iloc[i:i+width]).tolist())
            z.append(close_.iloc[i:i+width])
        x = np.array(x)
        return x,z


    def make_wave_dict(self,x,y,width):
        n_label = list(set(y))
        self.n_label = n_label
        wave_dict = {i:np.array([0.0 for j in range(width)]) for i in n_label}
        
        # クラス波形の総和
        for i in range(len(x)):
            wave_dict[y[i]] += x[i]
        
        # 平均クラス波形
        for i in range(len(y)):
            count_class = list(y).count(y[i])
            wave_dict[y[i]] /= count_class
            wave_dict[y[i]] = preprocessing.scale(wave_dict[y[i]])
        return wave_dict


    def learn_clustering(self,path_tpx,path_daw,width=20,stride=5,test_rate=0.8):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con['close']
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=test_rate)
        model = KMeans(n_clusters=self.n_cluster,random_state=self.random_state)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering2(self,close_,width=20,stride=5):
        x,_ = self.make_x_data(close_,width=width,stride=stride,test_rate=1.0)
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict


    def learn_clustering3(self,x,width=20,stride=5):
        model = KMeans(n_clusters=self.n_cluster)
        model.fit(x)
        self.model = model
        y = model.labels_
        wave_dict = self.make_wave_dict(x,y,width)
        self.wave_dict = wave_dict
    

    def show_class_wave(self):
        for i in range(self.n_cluster):
            print("--------------------")
            print("class :",i)
            plt.plot(self.wave_dict[i])
            plt.show()
            plt.clf()


    def predict(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z


    def predict2(self,df_con,stride=2,test_rate=1.0):
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred,z

    
    def return_y_pred(self,path_tpx,path_daw,stride=2,test_rate=1.0):
        df_con = self.make_df_con(path_tpx,path_daw)
        close_ = df_con["close"]
        x,z = self.make_x_data(close_,stride=stride,test_rate=test_rate)
        y_pred  = self.model.predict(x)
        return y_pred

    def encode(self, strategy, alpha, wave_dict):
        pass

class Simulation():


    def __init__(self):
        self.model = None
        self.accuracy_df = None
        self.trade_log = None
        self.pr_log = None
        self.MK = MakeTrainData
        self.ma_short =  5
        self.ma_long = 25
        self.wallet = 2500


    def simulate_routine(self, df,start_year=2021,end_year=2021,start_month=1,end_month=12,is_validate=False):

        df['ma_short'] = df['close'].rolling(self.ma_short).mean()
        df['ma_long']  = df['close'].rolling(self.ma_long).mean()
        df = df.iloc[self.ma_long:]
        df = self.return_split_df(df,start_year=start_year,end_year=end_year,start_month=start_month,end_month=end_month)

        self.df = df
        if not is_validate:
            pl = PlotTrade(df['close'],label='close')
            pl.add_plot(df['ma_short'],label='ma_short')
            pl.add_plot(df['ma_long'],label='ma_long')
        else:
            pl=None
        self.pr_log = pd.DataFrame(index=df.index[:-1])
        self.pr_log['reward'] = [0.0] * len(self.pr_log)
        self.pr_log['eval_reward'] = self.pr_log['reward'].tolist()

        return df,pl


    def set_for_online(self,x_check,y_):
        x_tmp = x_check
        y_tmp = y_
        current_date = x_tmp.index[0]
        acc_df = pd.DataFrame(index=x_tmp.index)
        acc_df['pred'] = [-1] * len(acc_df)
        return x_tmp, y_tmp, current_date, acc_df


        
    def learn_online(self,x_tmp,y_tmp,x_check,current_date,tmp_date):
        x_ = x_tmp[current_date<=x_tmp.index]
        x_ = x_[x_.index<tmp_date]
        y_ = y_tmp[current_date<=y_tmp.index]
        y_ = y_[y_.index<tmp_date]
        self.xgb_model = self.xgb_model.fit(x_,y_)
        predict_proba = self.xgb_model.predict_proba(x_check.astype(float))
        current_date = tmp_date

        return predict_proba, current_date


    def buy(self,df_con,x_check,i):
#   観測した始値が, 予測に反して上がっていた時, 買わない
        index_buy = df_con['close'].loc[x_check.index[i+1]]
        start_time = x_check.index[i+1]
        is_bought = True

        return index_buy, start_time, is_bought


    def sell(self,df_con,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate):
        index_sell = df_con['close'].loc[x_check.index[i+1]]
        end_time = x_check.index[i+1]
        prf += index_sell - index_buy
        prf_list.append(index_sell - index_buy)
        trade_count += 1
        is_bought = False
        if not is_validate:
            pl.add_span(start_time,end_time)
        else:
            pass

        return prf, trade_count, is_bought


    def hold(self,df_con,index_buy,total_eval_price,i):
        eval_price = df_con['close'].iloc[i] - index_buy
        total_eval_price += eval_price
        self.pr_log['eval_reward'].loc[df_con.index[i]] = total_eval_price
        return total_eval_price


# SELL の直後に BUY となる時, シミュレートできてない
    def return_grad(self, df, index, gamma=0, delta=0):
        grad_ma_short = df['ma_short'].iloc[index+1] - df['ma_short'].iloc[index]
        grad_ma_long  = df['ma_long'].iloc[index+1] - df['ma_long'].iloc[index]
        strategy = ''
        
        if grad_ma_long >= gamma:
            strategy = 'normal'
        elif grad_ma_long < delta:
            strategy = 'reverse'
        else:
            print("No such threshold")
        return strategy

    
    def make_df_con(self,path_tpx,path_daw):
        df_tpx = DataFramePreProcessing(path_tpx).load_df()
        df_daw = DataFramePreProcessing(path_daw,is_daw=True).load_df()
        daw_p = df_daw.pct_change()
        tpx_p = df_tpx.pct_change()
        tpx_p = tpx_p.rename(columns={'close':'pclose'})
        df_daw = df_daw.rename(columns={'dopen':'daw_close'})
        df_con = pd.concat([df_daw['daw_close'],df_tpx,daw_p['dclose'],tpx_p['pclose']],axis = 1,join='inner').astype(float)
        df_con = df_con.drop(df_con[ df_con['volume']==0].index)
        return df_con

    
    def make_check_data(self,path_tpx,path_daw):
        df_con = self.make_df_con(path_tpx,path_daw)
        mk = self.MK(df_con,test_rate=1.0)
        x_check, y_check, _, _ = mk.make_data()
        # self.ma_short = mk.ma_short
        # self.ma_long = mk.ma_long
        return x_check, y_check
    

    def calc_acc(self, acc_df, y_check):
        df = pd.DataFrame(columns = ['score','Up precision','Down precision','Up recall','Down recall','up_num','down_num'])
        acc_dict = {'TU':0,'FU':0,'TD':0,'FD':0}
    
        for i in range(len(acc_df)):
            
            label = acc_df['pred'].iloc[i]
            if label==-1:continue

            if y_check[i]==label:
                if label==0:
                    acc_dict['TD'] += 1
                else:#label = 1 : UP
                    acc_dict['TU'] += 1
            else:
                if label==0:
                    acc_dict['FD'] += 1
                else:
                    acc_dict['FU'] += 1

        df = self.calc_accuracy(acc_dict,df)
        return df


    def calc_accuracy(self,acc_dict,df):
        denom = 0
        for idx, key in enumerate(acc_dict):
            denom += acc_dict[key]
        
        try:
            TU = acc_dict['TU']
            FU = acc_dict['FU']
            TD = acc_dict['TD']
            FD = acc_dict['FD']
            score = (TU + TD)/(denom)
            prec_u = TU/(TU + FU)
            prec_d = TD/(TD + FD)
            recall_u = TU/(TU + FD)
            recall_d = TD/(TD + FU)
            up_num = TU+FD
            down_num = TD+FU
            col_list = [score,prec_u,prec_d,recall_u,recall_d,up_num,down_num]
            df.loc[0] = col_list
            return df
        except:
            print("division by zero")
            return None



# ここ間違ってる
    def return_split_df(self,df,start_year=2021,end_year=2021,start_month=1,end_month=12):
        df = df[df.index.year>=start_year]
        if start_year <= end_year:
            df = df[df.index.year<=end_year]
        if len(set(df.index.year))==1:
            df = df[df.index.month>=start_month]
            df = df[df.index.month<=end_month]
        else:
            df_tmp = df[df.index.year==start_year]
            last_year_index = df_tmp[df_tmp.index.month==start_month].index[0]
#             new_year_index = df[df.index.month==end_year].index[-1]
            df = df.loc[last_year_index:]
        return df


    def return_trade_log(self,prf,trade_count,prf_array,cant_buy):
        
        pr = (prf/self.wallet)*100
        log_dict = {
            'total_profit':prf,
            'profit rate':pr,
            'trade_count':trade_count,
            'max_profit':prf_array.max(),
            'min_profit':prf_array.min(),
            'mean_profit':prf_array.mean(),
            'cant_buy_count':cant_buy
            }
        df = pd.DataFrame(log_dict,index=[1])
        return df


    
    def get_accuracy(self):
        return self.accuracy_df


    def get_trade_log(self):
        return self.trade_log



    def simulate(self):
        pass


# simulate 済みを仮定
    def return_profit_rate(self,wallet=2500):
        self.pr_log['reward'] = self.pr_log['reward'].map(lambda x: x/wallet)
        self.pr_log['eval_reward'] = self.pr_log['eval_reward'].map(lambda x: x/wallet)
        return self.pr_log

class CeilSimulation(Simulation):


    def __init__(self,alpha=0.8,beta=0.2):
        super(CeilSimulation,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.MK = MakeTrainData


    def make_z_dict(self,df,stride=1,test_rate=1.0):
        z_dict = {}
        lc = LearnClustering()
        _, z_ = lc.make_x_data(df['close'],stride=stride,test_rate=test_rate)
        length = len(z_)

        for i in range(length):
            time_ = z_[i].index[-1]
            z_dict[time_] = z_[i]
        
        return z_dict


    def calc_ceil(self,z_,close_):
        L = z_.min()
        H = z_.max()
        ceil_ = (close_ - L)/(H - L)
        return ceil_


    def simulate(self,df, is_validate=False,is_online=False,start_year=2021,end_year=2021,start_month=1,end_month=12,
    is_observed=False):
        
        x_check,y_check,y_,df,pl = self.simulate_routine(df,start_year,end_year,start_month,end_month,'None',is_validate)
        z_dict = self.make_z_dict(df)
        
        length = len(df)
        
        prf_list = []
        is_bought = False
        index_buy = 0
        index_sell = 0
        prf = 0
        hold_day = 0
        trigger_count = 0
        is_trigger = False
        trade_count = 0
        total_eval_price = 0
        cant_buy = 0
        buy_count = 0
        sell_count = 0
        ceil_list = []

        for i in range(length-1):

            time_ = df.index[i]
            z_ = z_dict[time_]
            close_ = df['close'].iloc[i]
            ceil_ = self.calc_ceil(z_,close_)
            ceil_list.append(ceil_)

            total_eval_price = prf
            self.pr_log['reward'].loc[df.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df.index[i]] = total_eval_price

            # 底で買って, 天井で売る
            is_buy  = ceil_<self.beta
            is_sell = ceil_>self.alpha
            is_cant_buy = (is_observed and (df['open'].loc[x_check.index[i+1]] < df['close'].loc[x_check.index[i]]))

            
            if not is_bought:
                if is_cant_buy:
                    cant_buy += 1
                    continue
                elif is_buy:
                    index_buy, start_time, is_bought = self.buy(df,x_check,i)
                    buy_count += 1

            else:
                hold_day += 1
                if hold_day>=20:
                    trigger_count+=1
                    is_trigger = True

                if is_sell or is_trigger:
                    prf, trade_count, is_bought = self.sell(df,x_check,prf,index_buy,prf_list,trade_count,pl,start_time,i,is_validate)
                    hold_day = 0
                    is_trigger = False
                    sell_count += 1
                else:
                    total_eval_price = self.hold(df,index_buy,total_eval_price,i)
                    
            
            self.is_bought = is_bought
                
        
        if is_bought:
            index_sell = df['close'].loc[x_check.index[-1]] 
            prf += index_sell - index_buy
            prf_list.append(index_sell - index_buy)
            end_time = x_check.index[-1]
            trade_count+=1
            if not is_validate:
                pl.add_span(start_time,end_time)

        
        ceil_df = pd.DataFrame(ceil_list,columns={'ceil'},index=x_check.index[:-1])
        self.ceil_df = ceil_df
        self.pr_log['reward'].loc[df.index[-1]] = prf 
        self.pr_log['eval_reward'].loc[df.index[-1]] = total_eval_price
        prf_array = np.array(prf_list)
        self.y_check = y_check
        self.ceil_df = ceil_df
        log = self.return_trade_log(prf,trade_count,prf_array,cant_buy)
        self.trade_log = log

        if not is_validate:
            print(log)
            print("")
            print("trigger_count :",trigger_count)
            pl.show()


    def show_ceil_chart(self):
        plt.clf()
        chart_ = self.df['close']
        ceil_df = self.ceil_df.copy()
        scale = chart_.mean() * 0.9
        _, ax = plt.subplots(figsize=(20, 6))
        ax.plot(chart_.iloc[:-1],label='close')
        ax.plot(ceil_df['ceil']*scale,label='ceil')
        plt.grid()
        plt.show()

class TechnicalSimulation(Simulation):
    
    
    def __init__(self,ma_short=5, ma_long=25, hold_day=5, year=2021):
        super(TechnicalSimulation,self).__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.hold_day = hold_day
        self.year = year
        
    
    def is_buyable(self, short_line, long_line, index_):
#         1=<index<=len-1 仮定
        long_is_upper = long_line.iloc[index_-1]>short_line.iloc[index_-1]
        long_is_lower = long_line.iloc[index_]<=short_line.iloc[index_]
        buyable = long_is_upper and long_is_lower
        return buyable
    
    
    def is_sellable(self, short_line, long_line, index_):
        long_is_lower = long_line.iloc[index_-1]<short_line.iloc[index_-1]
        long_is_upper = long_line.iloc[index_]>=short_line.iloc[index_]
        sellable = long_is_upper and long_is_lower
        return sellable

        
        
    def simulate(self,df,is_validate=False,start_year=2021,end_year=2021,start_month=1,end_month=12):
        
        df,pl = self.simulate_routine(df,start_year,end_year,start_month,end_month)
        
        prf_list = []
        is_bought = False
        index_buy = 0
        prf = 0
        trade_count = 0
        eval_price = 0
        total_eval_price = 0
        short_line = df['ma_short']
        long_line = df['ma_long']
        length = len(df)

        for i in range(1,length):
            
            total_eval_price = prf
            self.pr_log['reward'].loc[df.index[i]] = prf 
            self.pr_log['eval_reward'].loc[df.index[i]] = total_eval_price
            if not is_bought:
                
                if self.is_buyable(short_line,long_line,i):
                    index_buy = df['close'].iloc[i]
                    is_bought = True
                    start_time = df.index[i]
                    hold_count_day = 0
                else:
                    continue
            
            
            else:
                
                if self.is_sellable(short_line,long_line,i) or hold_count_day==self.hold_day:
                    index_cell = df['close'].iloc[i]
                    end_time = df.index[i]
                    prf += index_cell - index_buy
                    prf_list.append(index_cell - index_buy)
                    total_eval_price = prf
                    self.pr_log['reward'].loc[df.index[i]] = prf 
                    self.pr_log['eval_reward'].loc[df.index[i]] = total_eval_price
                    trade_count+=1
                    is_bought = False
                    hold_count_day = 0
                    pl.add_span(start_time,end_time)
                else:
                    hold_count_day+=1
                    eval_price = df['close'].iloc[i] - index_buy
                    total_eval_price += eval_price
                    self.pr_log['eval_reward'].loc[df.index[i]] = total_eval_price
                    
        
        if is_bought:
            end_time = df['close'].index[-1]
            index_sell = df['close'].iloc[-1]
            pl.add_span(start_time,end_time)
            eval_price = index_sell - index_buy
            prf += eval_price
            prf_list.append(prf)
            total_eval_price += eval_price
            self.pr_log['eval_reward'].loc[df.index[-1]] = total_eval_price
        
        prf_array = np.array(prf_list)
        log = self.return_trade_log(prf,trade_count,prf_array,0)
        self.trade_log = log

        if not is_validate:        
            print(log)
            print("")
            pl.show()    


