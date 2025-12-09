import os
import gym
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# --- 데이터 처리 함수 ---

def download_data(symbols, start_date, end_date):
    """Yahoo Finance에서 데이터 다운로드"""
    # yfinance 업데이트로 인해 auto_adjust=True 권장, 혹은 Adj Close 명시적 사용
    price_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    price_data['Cash'] = 1.0
    return price_data

def calculate_simple_returns(price_data):
    """단순 수익률 계산"""
    simple_returns = price_data.pct_change().dropna()
    simple_returns['Cash'] = 0.0
    return simple_returns

def calculate_volatility_metrics(kospi_data):
    """KOSPI 지수 기반 변동성 지표 계산"""
    kospi_returns = kospi_data['Adj Close'].pct_change().dropna()
    
    vol20 = kospi_returns.rolling(window=20).std()
    vol60 = kospi_returns.rolling(window=60).std()
    
    vol20_vol60 = vol20 / vol60
    
    # 정규화 (Expanding window)
    vol20 = (vol20 - vol20.expanding().mean()) / vol20.expanding().std()
    vol20_vol60 = (vol20_vol60 - vol20_vol60.expanding().mean()) / vol20_vol60.expanding().std()
    
    return vol20.fillna(0), vol20_vol60.fillna(0)

# --- Gym 환경 정의 (DSR Reward) ---

class PortfolioEnvDSR(gym.Env):
    """
    Differential Sharpe Ratio (DSR)를 보상으로 사용하는 포트폴리오 환경
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, returns_df, kospi_data, vkospi_data, initial_cash=100000000, lookback_period=10):
        super(PortfolioEnvDSR, self).__init__()
        
        self.df = returns_df
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.n_assets = returns_df.shape[1]
        
        # 거시 경제 지표 준비
        vol20, vol20_vol60 = calculate_volatility_metrics(kospi_data)
        # 거래일 차이로 인한 인덱스 불일치 방지: 자산 수익률 인덱스에 맞춰 재정렬
        self.vol20 = vol20.reindex(self.df.index).fillna(method='ffill').fillna(0)
        self.vol20_vol60 = vol20_vol60.reindex(self.df.index).fillna(method='ffill').fillna(0)
        self.vkospi = vkospi_data['Adj Close'].reindex(self.df.index).fillna(method='ffill').fillna(0)

        # Action Space: 자산 비중 (Cash 제외 n-1개)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets - 1,), dtype=np.float32)
        
        # Observation Space: (자산 수익률 + 3개 거시 지표) x lookback_period
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets + 3, lookback_period), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.current_step = self.lookback_period
        self.portfolio_value = self.initial_cash
        self.total_asset_log = [self.portfolio_value]
        self.portfolio_weights_log = []
        self.daily_returns = []
        self.dates_log = []

        # DSR 계산을 위한 이동 평균 변수 초기화
        self.A = np.zeros(self.n_assets) # 1차 모멘트 (수익률 평균)
        self.B = np.zeros(self.n_assets) # 2차 모멘트 (수익률 제곱의 평균)
        self.eta = 1 / 252 # 감쇠 계수 (1년 기준)

        obs = self._next_observation()
        return obs

    def step(self, action):
        # Action 정규화 및 Cash 비중 계산
        action = np.array(action)
        if action.sum() > 1:
            action /= action.sum()
        weights = np.append(action, 1 - action.sum())

        # 포트폴리오 가치 업데이트
        current_returns = self.df.iloc[self.current_step]
        current_portfolio_value = self.portfolio_value
        asset_contributions = 1 + current_returns
        new_portfolio_value = np.dot(weights, asset_contributions * current_portfolio_value)

        # --- DSR(Differential Sharpe Ratio) 보상 계산 로직 ---
        # 개별 자산별 DSR 업데이트가 아닌 포트폴리오 전체 수익률에 대한 DSR을 계산하는 것이 일반적이나,
        # 제공해주신 코드는 자산별 모멘트를 업데이트하고 합산하는 방식을 사용 중이므로 해당 로직 유지.
        
        delta_A_t = current_returns - self.A
        delta_B_t = current_returns ** 2 - self.B

        # D_t 계산 (분모가 0이 되는 것 방지)
        denom = (np.abs(self.B - self.A ** 2) ** (3 / 2))
        D_t = np.zeros_like(self.B)
        mask = denom != 0
        D_t[mask] = ((self.B[mask] * delta_A_t[mask]) - (0.5 * self.A[mask] * delta_B_t[mask])) / denom[mask]
        
        # 이동 평균 업데이트
        self.A += self.eta * delta_A_t
        self.B += self.eta * delta_B_t

        reward = np.sum(D_t)
        # -----------------------------------------------------

        # 로그 기록
        self.portfolio_value = new_portfolio_value
        self.total_asset_log.append(self.portfolio_value)
        self.portfolio_weights_log.append(weights.tolist())
        
        daily_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value if current_portfolio_value != 0 else 0
        self.daily_returns.append(daily_return)
        
        self.dates_log.append(self.df.index[self.current_step].strftime('%Y-%m-%d'))
        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        
        next_observation = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return next_observation, reward, done, {}

    def _next_observation(self):
        start = self.current_step - self.lookback_period
        end = self.current_step
        
        # 자산 수익률
        frame = self.df.iloc[start:end].T.values
        
        # 거시 지표 슬라이싱
        vol20 = self.vol20.iloc[start:end].values
        vol20_vol60 = self.vol20_vol60.iloc[start:end].values
        vkospi = self.vkospi.iloc[start:end].values
        
        # 차원 결합
        additional_features = np.array([vol20, vol20_vol60, vkospi])
        frame = np.vstack((frame, additional_features))
        
        return frame

def make_env(returns_df, kospi_data, vkospi_data, initial_cash, lookback_period):
    def _init():
        return PortfolioEnvDSR(returns_df, kospi_data, vkospi_data, initial_cash=initial_cash, lookback_period=lookback_period)
    return _init

# --- 메인 실행 함수 ---

def main():
    # 1. 설정 및 경로
    ASSETS = ['005930.KS', '000660.KS', '105560.KS', '005380.KS', '035420.KS',
              '051910.KS', '000100.KS', '047050.KS', '042700.KS', '005490.KS',
              '000810.KS', '011200.KS', '001040.KS', '017670.KS', '006800.KS', 'Cash']
    
    START_DATE = '2005-01-01'
    END_DATE = '2024-01-01'
    
    # 로컬 경로 설정 (저장소 구조에 맞게)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data') # 상위 폴더의 data
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models_kor')
    RESULTS_SAVE_PATH = os.path.join(BASE_DIR, 'results_kor')
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True) # Data 폴더가 없으면 생성 (여기에 csv 넣어야 함)

    VKOSPI_FILE = os.path.join(DATA_PATH, 'vkospi_data.csv')

    # 하이퍼파라미터
    LOOKBACK_PERIOD = 10
    TRAINING_TIMESTEPS = 100000
    N_ENVS = 10
    N_STEPS = 256
    BATCH_SIZE = 126
    N_EPOCHS = 8
    GAMMA = 0.9
    GAE_LAMBDA = 0.9
    CLIP_RANGE = 0.25
    LEARNING_RATE = 3e-4
    
    # 슬라이딩 윈도우
    START_YEAR = 2006
    END_YEAR = 2023
    PERIOD_LENGTH = 7
    TRAINING_PERIOD = 5
    VALIDATION_PERIOD = 1
    NUM_SEEDS = 5

    # 2. 데이터 로드
    print("데이터 다운로드 중 (KOSPI Assets)...")
    price_data = download_data(ASSETS[:-1], START_DATE, END_DATE) # Cash 제외하고 다운
    if 'CASH' in price_data.columns:
        price_data = price_data.drop(columns=['CASH'])
    
    returns_data = calculate_simple_returns(price_data)
    
    print("데이터 다운로드 중 (KOSPI Index)...")
    kospi_data = yf.download('^KS11', start=START_DATE, end=END_DATE)
    
    print(f"VKOSPI 데이터 로드 중: {VKOSPI_FILE}")
    if not os.path.exists(VKOSPI_FILE):
        print(f"[오류] {VKOSPI_FILE} 파일을 찾을 수 없습니다.")
        print("data 폴더에 vkospi_data.csv 파일을 넣어주세요.")
        return

    vkospi_data = pd.read_csv(VKOSPI_FILE, index_col='Date', parse_dates=True)
    vkospi_data = vkospi_data.rename(columns={'Close': 'Adj Close'})
    # 날짜 인덱스 맞추기 (ffill)
    vkospi_data = vkospi_data.reindex(returns_data.index).fillna(method='ffill')

    # 3. 훈련 루프
    print("=== KOSPI 포트폴리오(DSR) 훈련 시작 ===")
    
    full_portfolio_value_log = []
    full_daily_returns = []
    full_daily_weights = []
    full_dates = []

    for start in range(START_YEAR, END_YEAR - PERIOD_LENGTH + 2):
        # 훈련 기간 설정 (오프셋 조정)
        train_start = pd.to_datetime(f"{start}-01-01") - pd.DateOffset(days=2*LOOKBACK_PERIOD + 30)
        train_end = f"{start + TRAINING_PERIOD - 1}-12-31"
        validation_start = pd.to_datetime(f"{start + TRAINING_PERIOD}-01-01") - pd.DateOffset(days=2*LOOKBACK_PERIOD + 30)
        validation_end = f"{start + TRAINING_PERIOD}-12-31"
        test_start = pd.to_datetime(f"{start + TRAINING_PERIOD + VALIDATION_PERIOD}-01-01") - pd.DateOffset(days=2*LOOKBACK_PERIOD + 30)
        test_end = f"{start + PERIOD_LENGTH - 1}-12-31"

        print(f"\n--- 윈도우: {start} ~ {start + PERIOD_LENGTH - 1} ---")
        
        train_data = returns_data[train_start:train_end]
        validation_data = returns_data[validation_start:validation_end]
        test_data = returns_data[test_start:test_end]

        # 초기 자본금 설정 (첫 해는 1억, 이후는 이전 윈도우 최종값)
        if start == START_YEAR:
            initial_cash = 100000000
        else:
            initial_cash = full_portfolio_value_log[-1]

        best_agent = None
        best_performance = -np.inf

        # 시드별 훈련
        for seed in range(NUM_SEEDS):
            set_random_seed(seed)
            # 환경 생성 (VKOSPI 데이터 전달)
            env = DummyVecEnv([make_env(train_data, kospi_data, vkospi_data, initial_cash, LOOKBACK_PERIOD)])
            
            model = PPO("MlpPolicy", env, verbose=0, seed=seed,
                        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
                        gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
                        learning_rate=LEARNING_RATE)
            
            model.learn(total_timesteps=TRAINING_TIMESTEPS)

            # 검증
            val_env = DummyVecEnv([make_env(validation_data, kospi_data, vkospi_data, initial_cash, LOOKBACK_PERIOD)])
            obs = val_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = val_env.step(action)
                total_reward += reward[0]

            if total_reward > best_performance:
                best_performance = total_reward
                best_agent = model
                print(f"  Best Model Updated (Seed {seed}, Reward: {total_reward:.2f})")

        # 모델 저장
        model_name = f'ppo_dsr_kor_{start}_{start + PERIOD_LENGTH - 1}.zip'
        best_agent.save(os.path.join(MODEL_SAVE_PATH, model_name))

        # 테스트 (Backtest)
        print("  Testing Best Agent...")
        loaded_model = PPO.load(os.path.join(MODEL_SAVE_PATH, model_name))
        
        # 테스트 시 초기 자본금은 이전 로그에서 가져오거나 초기화
        test_env = DummyVecEnv([make_env(test_data, kospi_data, vkospi_data, initial_cash, LOOKBACK_PERIOD)])
        obs = test_env.reset()
        done = False
        
        # 테스트 환경에서 실제 기간 시작점 찾기 (Lookback 제외)
        # DummyVecEnv는 envs 리스트를 가짐
        raw_env = test_env.envs[0] 
        
        # 로그 수집용 리스트 초기화 (환경 내부 로그 사용)
        
        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, done, _ = test_env.step(action)
            # VecEnv에서 반환되는 done은 배열이므로 첫 원소 사용
            done = bool(done[0])

        # 환경 내부 로그 추출
        full_portfolio_value_log.extend(raw_env.total_asset_log[1:])
        full_daily_returns.extend(raw_env.daily_returns)
        full_daily_weights.extend(raw_env.portfolio_weights_log)
        full_dates.extend(raw_env.dates_log)

        print(f"  Test Completed. Final Value: {raw_env.portfolio_value:,.0f} KRW")

    # 4. 결과 저장
    print("\n결과 저장 중...")
    min_len = min(len(full_dates), len(full_portfolio_value_log), len(full_daily_returns), len(full_daily_weights))
    
    data = {
        'Date': full_dates[:min_len],
        'Portfolio Value': full_portfolio_value_log[:min_len],
        'Daily Returns': full_daily_returns[:min_len],
        'Daily Weights': full_daily_weights[:min_len]
    }
    
    result_df = pd.DataFrame(data)
    result_df.set_index('Date', inplace=True)
    
    result_df.to_csv(os.path.join(RESULTS_SAVE_PATH, 'PPO_DSR_KOR_results.csv'))
    print("완료.")

if __name__ == "__main__":
    main()
