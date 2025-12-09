import os
import gym
import random
from gym import spaces
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# --- 데이터 처리 함수 ---

def download_data(symbols, start_date, end_date):
    """지정된 심볼과 기간으로 가격 데이터를 다운로드합니다."""
    price_data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    price_data['Cash'] = 1.0
    return price_data

def calculate_simple_returns(price_data):
    """가격 데이터로부터 일일 단순 수익률을 계산합니다."""
    simple_returns = price_data.pct_change().dropna()
    simple_returns['Cash'] = 0.0
    return simple_returns

def calculate_volatility_metrics(sp500_data):
    """S&P 500 데이터로부터 변동성 지표를 계산합니다."""
    sp500_returns = sp500_data['Adj Close'].pct_change().dropna()
    vol20 = sp500_returns.rolling(window=20).std()
    vol60 = sp500_returns.rolling(window=60).std()
    vol20_vol60 = vol20 / vol60
    
    # 정규화
    vol20 = (vol20 - vol20.expanding().mean()) / vol20.expanding().std()
    vol20_vol60 = (vol20_vol60 - vol20_vol60.expanding().mean()) / vol20_vol60.expanding().std()
    
    return vol20.fillna(0), vol20_vol60.fillna(0)

# --- Gym 환경 정의 ---

class PortfolioEnvDSR(gym.Env):
    """포트폴리오 관리를 위한 커스텀 Gym 환경 (DSR 보상 아님 - 단순 수익률 보상)"""
    metadata = {'render.modes': ['console']}

    def __init__(self, returns_df, sp500_data, vix_data, initial_cash=100000, lookback_period=60, **hyperparameters):
        super(PortfolioEnvDSR, self).__init__()
        
        self.df = returns_df
        self.vol20, self.vol20_vol60 = calculate_volatility_metrics(sp500_data)
        self.vix = (vix_data - vix_data.expanding().mean()) / vix_data.expanding().std().fillna(0) # VIX 정규화
        
        self.initial_cash = initial_cash
        self.lookback_period = lookback_period
        self.n_assets = returns_df.shape[1]
        
        # Action: n-1개 자산의 비중 (마지막 'Cash' 비중은 1에서 뺌)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets - 1,), dtype=np.float32)
        
        # Observation: (자산 수익률 + 3개 거시 지표) x lookback_period
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets + 3, lookback_period), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.current_step = self.lookback_period
        self.portfolio_value = self.initial_cash
        self.total_asset_log = [self.portfolio_value]
        self.portfolio_weights_log = []
        self.daily_returns = []
        self.dates_log = []

        obs = self._next_observation()
        return obs

    def step(self, action):
        # 액션(포트폴리오 비중) 정규화
        action = np.array(action)
        if action.sum() > 1:  # 합이 1을 넘으면 정규화
            action /= action.sum()
        
        # Cash 비중 추가
        weights = np.append(action, 1 - action.sum())

        current_returns = self.df.iloc[self.current_step]
        current_portfolio_value = self.portfolio_value
        
        # 포트폴리오 가치 계산
        asset_contributions = 1 + current_returns
        new_portfolio_value = np.dot(weights, asset_contributions * current_portfolio_value)

        # 보상: 포트폴리오 가치의 단순 변화량 (논문의 'Reward to Return' 전략)
        reward = new_portfolio_value - current_portfolio_value

        # 로그 기록
        self.portfolio_value = new_portfolio_value
        self.total_asset_log.append(self.portfolio_value)
        # 기록을 누적해야 백테스트 결과와 길이가 맞음
        self.portfolio_weights_log.append(weights.tolist())
        daily_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value if current_portfolio_value != 0 else 0
        self.daily_returns.append(daily_return)
        self.dates_log.append(self.df.index[self.current_step].strftime('%Y-%m-%d'))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        next_observation = self._next_observation() if not done else np.zeros(self.observation_space.shape)

        return next_observation, reward, done, {}

    def _next_observation(self):
        """현재 스텝 기준
        - [lookback_period]일간의 자산 수익률
        - [lookback_period]일간의 3가지 거시 지표
        """
        start = self.current_step - self.lookback_period
        end = self.current_step

        frame = self.df.iloc[start:end].T.values
        vol20 = self.vol20.iloc[start:end].values
        vol20_vol60 = self.vol20_vol60.iloc[start:end].values
        vix = self.vix.iloc[start:end].values
        
        # (n_assets + 3, lookback_period)
        additional_features = np.array([vol20, vol20_vol60, vix])
        frame = np.vstack((frame, additional_features))
        return frame

    def render(self, mode='console'):
        print(f"Step: {self.current_step}, Portfolio Value: {self.total_asset_log[-1]:.2f}")

    def close(self):
        pass

def make_env(returns_df, sp500_data, vix_data, initial_cash, lookback_period):
    """환경 생성 헬퍼 함수"""
    def _init():
        # VIX 데이터를 환경 생성자에 전달
        return PortfolioEnvDSR(returns_df, sp500_data, vix_data, initial_cash=initial_cash, lookback_period=lookback_period)
    return _init

# --- 메인 실행 함수 ---

def main():
    """메인 스크립트 실행 함수"""

    # --- 1. 설정 및 하이퍼파라미터 ---
    
    # 데이터 설정
    ASSETS = ['XLB', 'XLI', 'XLY', 'XLP', 'XLV', 'XLF', 'XLK', 'XLU', 'XLE', 'Cash']
    START_DATE = '2005-01-01'
    END_DATE = '2023-12-31'

    # 경로 설정 (상대 경로)
    MODEL_SAVE_PATH = './models'
    RESULTS_SAVE_PATH = './results'
    
    # 폴더 생성
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    
    # 환경 하이퍼파라미터
    LOOKBACK_PERIOD = 60
    
    # PPO 하이퍼파라미터 (논문과 동일하게 설정 시)
    TRAINING_TIMESTEPS = 100000  # 학습할 총 타임스텝 수
    N_ENVS = 10                  # 병렬 환경 수
    N_STEPS = 256                # 업데이트당 스텝 수
    BATCH_SIZE = 126             # 미니배치 크기
    N_EPOCHS = 8                 # 에포크 수
    GAMMA = 0.9                  # 할인율
    GAE_LAMBDA = 0.9             # GAE 람다
    CLIP_RANGE = 0.25            # 클리핑 범위
    LEARNING_RATE = 3e-4         # 학습률

    # 슬라이딩 윈도우 설정
    START_YEAR = 2006
    END_YEAR = 2023
    PERIOD_LENGTH = 7
    TRAINING_PERIOD = 5
    VALIDATION_PERIOD = 1
    NUM_SEEDS = 5  # 각 윈도우당 시도할 시드 수
    
    # --- 2. 데이터 로드 및 전처리 ---
    
    print("데이터 다운로드 중...")
    price_data = download_data(ASSETS, START_DATE, END_DATE)
    if 'CASH' in price_data.columns:
        price_data = price_data.drop(columns=['CASH'])
    
    returns_data = calculate_simple_returns(price_data)
    
    sp500_data = yf.download('^GSPC', start=START_DATE, end=END_DATE)
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE)['Adj Close']
    print("데이터 로드 완료.")

    # --- 3. 훈련 (Sliding Window) ---
    
    print("=== 모델 훈련 시작 ===")
    
    # 훈련 루프
    for start in range(START_YEAR, END_YEAR - PERIOD_LENGTH + 2):
        train_start = pd.to_datetime(f"{start}-01-01") - pd.DateOffset(days=2*LOOKBACK_PERIOD)
        train_end = f"{start + TRAINING_PERIOD - 1}-12-31"
        
        validation_start = pd.to_datetime(f"{start + TRAINING_PERIOD}-01-01") - pd.DateOffset(days=2*LOOKBACK_PERIOD)
        validation_end = f"{start + TRAINING_PERIOD}-12-31"
        
        print(f"\n--- 훈련 기간: {train_start.date()} ~ {validation_end.date()} ---")

        train_data = returns_data[train_start:train_end]
        validation_data = returns_data[validation_start:validation_end]
        
        initial_cash = 100000  # 훈련/검증 시 초기 자본금은 항상 10만으로 고정

        best_agent = None
        best_performance = -np.inf

        for seed in range(NUM_SEEDS):
            print(f"  시드 {seed+1}/{NUM_SEEDS} 훈련 중...")
            set_random_seed(seed)
            env = DummyVecEnv([make_env(train_data, sp500_data, vix_data, initial_cash, LOOKBACK_PERIOD)])
            
            model = PPO("MlpPolicy", env, verbose=0,  # verbose=1로 변경 시 상세 로그 출력
                        n_steps=N_STEPS,
                        batch_size=BATCH_SIZE,
                        n_epochs=N_EPOCHS,
                        gamma=GAMMA,
                        gae_lambda=GAE_LAMBDA,
                        clip_range=CLIP_RANGE,
                        learning_rate=LEARNING_RATE,
                        seed=seed)
            
            model.learn(total_timesteps=TRAINING_TIMESTEPS)

            # 검증 세트로 평가
            validation_env = DummyVecEnv([make_env(validation_data, sp500_data, vix_data, initial_cash, LOOKBACK_PERIOD)])
            obs = validation_env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = validation_env.step(action)
                total_reward += reward[0] # VecEnv이므로 [0] 인덱스 접근
                done = done[0]  # VecEnv가 bool 배열을 반환하므로 스칼라로 변환

            print(f"  시드 {seed+1} 검증 완료. 총 보상: {total_reward:.2f}")

            if total_reward > best_performance:
                best_performance = total_reward
                best_agent = model
                print(f"  *** 새 최고 성능 모델 발견 (보상: {best_performance:.2f}) ***")

        # 현재 기간의 최고 모델 저장
        best_model_save_path = os.path.join(MODEL_SAVE_PATH, f'model_{start}_{start + PERIOD_LENGTH-1}.zip')
        best_agent.save(best_model_save_path)
        print(f"모델 저장 완료: {best_model_save_path}")

    print("=== 모든 기간 훈련 완료 ===")


    # --- 4. 테스트 (Backtesting) ---
    
    print("\n=== 백테스팅 시작 ===")
    
    full_portfolio_value_log = []
    full_daily_returns = []
    full_daily_weights = []
    full_dates = []

    initial_cash_backtest = 100000 # 백테스트 시작 시점의 자본
    
    for start in range(START_YEAR, END_YEAR - PERIOD_LENGTH + 2):
        
        # 테스트 기간 설정 (훈련/검증과 겹치지 않게)
        test_start_date = f"{start + TRAINING_PERIOD + VALIDATION_PERIOD}-01-01"
        test_end_date = f"{start + TRAINING_PERIOD + VALIDATION_PERIOD}-12-31"

        # 데이터가 1년치를 정확히 포함하도록 오프셋 조정
        test_start = pd.to_datetime(test_start_date) - pd.DateOffset(days=2*LOOKBACK_PERIOD)
        test_end = pd.to_datetime(test_end_date)
        
        test_data = returns_data[test_start:test_end]
        
        print(f"\n--- 테스트 기간: {test_start_date} ~ {test_end_date} ---")

        # 해당 기간에 훈련된 모델 로드
        model_path = os.path.join(MODEL_SAVE_PATH, f'model_{start}_{start + PERIOD_LENGTH - 1}.zip')
        if not os.path.exists(model_path):
            print(f"  경고: 모델 파일을 찾을 수 없습니다. {model_path}")
            continue
            
        loaded_model = PPO.load(model_path)

        # 테스트 환경 생성
        # 백테스팅 시 초기 자본금은 이전 기간의 최종 금액을 이어받음
        if start > START_YEAR:
            initial_cash_backtest = full_portfolio_value_log[-1]
            
        test_env = DummyVecEnv([make_env(test_data, sp500_data, vix_data, initial_cash_backtest, LOOKBACK_PERIOD)])
        obs = test_env.reset()
        
        # 실제 테스트 기간(1년) 동안만 루프 실행
        # test_data에는 lookback을 위한 과거 데이터가 포함되어 있으므로, 실제 시작점을 찾아야 함
        actual_test_start_index = test_data.index.get_loc(test_start_date, method='bfill')
        
        # env.current_step을 실제 테스트 시작점으로 강제 설정
        test_env.envs[0].current_step = actual_test_start_index
        obs = test_env.envs[0]._next_observation().reshape(1, test_env.observation_space.shape[0], test_env.observation_space.shape[1]) # obs 수동 갱신
        
        done = False
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            # done은 VecEnv 래퍼에 의해 bool 배열일 수 있음
            done = done[0] 

        # 테스트 환경에서 로그 수집
        env_logs = test_env.envs[0]
        full_portfolio_value_log.extend(env_logs.total_asset_log[1:]) # 초기값 제외
        full_daily_returns.extend(env_logs.daily_returns)
        full_daily_weights.extend(env_logs.portfolio_weights_log)
        full_dates.extend(env_logs.dates_log)

        print(f"  테스트 완료. 최종 포트폴리오 가치: {full_portfolio_value_log[-1]:.2f}")

    print("=== 모든 기간 백테스팅 완료 ===")
    
    # --- 5. 결과 저장 ---
    
    print("\n결과 저장 중...")
    
    # 데이터프레임 생성
    # 참고: 로그 길이에 차이가 있을 수 있으므로, 가장 짧은 길이에 맞추거나 dict(list)로 생성
    min_len = min(len(full_dates), len(full_portfolio_value_log), len(full_daily_returns), len(full_daily_weights))

    data = {
        'Date': full_dates[:min_len],
        'Portfolio Value': full_portfolio_value_log[:min_len],
        'Daily Returns': full_daily_returns[:min_len],
        'Daily Weights': full_daily_weights[:min_len]
    }

    result_df = pd.DataFrame(data)
    result_df.set_index('Date', inplace=True)

    # CSV 파일로 저장
    result_csv_path = os.path.join(RESULTS_SAVE_PATH, 'PPO_return_reward_results.csv')
    result_df.to_csv(result_csv_path)

    print(f"백테스팅 결과가 {result_csv_path} 에 저장되었습니다.")
    
    # --- 6. 간단한 시각화 (선택 사항) ---
    try:
        plt.figure(figsize=(12, 6))
        result_df['Portfolio Value'].plot()
        plt.title('PPO (Return Reward) Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plot_path = os.path.join(RESULTS_SAVE_PATH, 'PPO_performance_plot.png')
        plt.savefig(plot_path)
        print(f"성능 그래프가 {plot_path} 에 저장되었습니다.")
    except Exception as e:
        print(f"그래프 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
