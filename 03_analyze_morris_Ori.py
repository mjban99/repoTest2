import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from SALib.analyze import morris as analyze_morris

# ==============================================================================
# 1. 문제 정의 및 샘플 데이터 로드
# ==============================================================================
try:
    X_df = pd.read_csv('morris_samples.csv', index_col=0)
    X = X_df.values
    var_names = X_df.columns.tolist()
    num_vars = len(var_names)
    print(f">>> 입력 데이터(X) 로드 완료. 변수 갯수: {num_vars}개, 총 {len(X)}번의 Run")
except FileNotFoundError:
    print("Error: 'morris_samples.csv' 파일이 없습니다.")
    exit()

file_path_boundary = 'data/WQ_Nak/WQ_reaction_params_new_boundary.csv'
df_boundary = pd.read_csv(file_path_boundary)
param_names = df_boundary.columns[1:].tolist()
lower_bounds = df_boundary.iloc[0, 1:].values.astype(float)
upper_bounds = df_boundary.iloc[1, 1:].values.astype(float)

print(f"NumberOfVariables: {num_vars}")
print(f"FirstVar: {param_names[0]} (Min: {lower_bounds[0]}, Max: {upper_bounds[0]})")

problem = {
    'num_vars': num_vars,
    'names': param_names,
    'bounds': [[l, u] for l, u in zip(lower_bounds, upper_bounds)]
}

# ==============================================================================
# 2. 결과 수집 (시계열 데이터로 수집)
# ==============================================================================
Y_time_series = [] # 전체 배열을 담을 리스트
valid_indices = []
J_day_array = None

target_variable = 'Q'
target_loc_idx = 9

print(f">>> 결과 값(Y) 시계열 데이터 수집 중...")

for run_id in range(len(X)):
    folder_path = f'./data/WQ_Nak/SC01_morris_run_{run_id}'
    if target_variable == 'Q':
        result_path = f'{folder_path}/tri_Q_runoff_cal.pkl'
    else:
        result_path = f'{folder_path}/c_j_output.pkl'
    j_day_path = f'{folder_path}/output_J_day.pkl'
    
    try:
        if not os.path.exists(result_path):
            continue

        with open(result_path, 'rb') as fp:
            data = pickle.load(fp)
            
        # J_day 데이터 로드 (한 번만 수행)
        if J_day_array is None and os.path.exists(j_day_path):
            with open(j_day_path, 'rb') as fp:
                J_day_array = np.array(pickle.load(fp))

        if target_variable == 'Q':
            val_data = np.array(data['Namgang']) 
        else:
            val_data = np.array(data[target_variable])[:, target_loc_idx]
        
        # 배열 내에 NaN/Inf가 하나도 없을 때만 유효한 Run으로 취급
        if np.isfinite(val_data).all():
            Y_time_series.append(val_data)
            valid_indices.append(run_id)
        else:
            print(f"Run {run_id}: 값 이상 (NaN/Inf 포함)")

    except Exception as e:
        print(f"Run {run_id} 에러: {e}")
        continue

# ------------------------------------------------------------------------------
# 💡 [작성하신 핵심 부분] 온전한 세트만 남기고 자투리 데이터 잘라내기
# ------------------------------------------------------------------------------
X_temp = X[valid_indices]
Y_matrix_temp = np.array(Y_time_series) # 2차원 행렬: 형태(유효 Run수, 타임스텝 수)

print(f"\n>>> 총 {len(X)}개 중 {len(Y_matrix_temp)}개의 유효 Run 결과를 확보했습니다.")

# 1개 세트당 필요한 Run 개수
num_steps = num_vars + 1 
# 온전한 세트 개수 계산
num_trajectories = len(Y_matrix_temp) // num_steps 
# 유지할 딱 떨어지는 데이터 개수
keep_count = num_trajectories * num_steps 

# 딱 떨어지는 만큼만 잘라서 덮어쓰기! (Y_matrix는 2차원이므로 행 기준으로 자름)
X_filtered = X_temp[:keep_count]
Y_matrix_filtered = Y_matrix_temp[:keep_count, :]

num_runs, num_timesteps = Y_matrix_filtered.shape
print(f">>> 이 중 모리스 분석에 사용될 완벽한 세트: {keep_count}개 ({num_trajectories}세트)")
print(f">>> 추출된 시계열(날짜) 길이: {num_timesteps}일")

# ==============================================================================
# 3. 타임스텝별 Morris 분석 실행
# ==============================================================================
print("\n>>> 타임스텝별 Morris 분석 수행 중... (시간이 소요될 수 있습니다)")

mu_list, mu_star_list, sigma_list = [], [], []

for t in range(num_timesteps):
    # 특정 타임스텝(t)에 대한 모든 Run의 결과값 (1D 배열)
    Y_t = Y_matrix_filtered[:, t]
    
    # 해당 날짜에 대한 Morris 분석
    Si_t = analyze_morris.analyze(problem, X_filtered, Y_t, conf_level=0.95, print_to_console=False)
    
    mu_list.append(Si_t['mu'])
    mu_star_list.append(Si_t['mu_star'])
    sigma_list.append(Si_t['sigma'])

    # 진행률 표시
    if (t + 1) % 50 == 0 or (t + 1) == num_timesteps:
        print(f"  -> {t + 1}/{num_timesteps} 타임스텝 처리 완료")

# ==============================================================================
# 4. 분석 결과 DataFrame 변환 및 CSV 저장
# ==============================================================================
print("\n>>> 분석 결과를 시계열 CSV로 저장하는 중...")

# J_day 데이터가 없으면 0, 1, 2... 식의 인덱스 사용
index_col = J_day_array if J_day_array is not None else np.arange(num_timesteps)

df_mu = pd.DataFrame(mu_list, columns=param_names, index=index_col)
df_mu_star = pd.DataFrame(mu_star_list, columns=param_names, index=index_col)
df_sigma = pd.DataFrame(sigma_list, columns=param_names, index=index_col)

# 인덱스명 지정
df_mu.index.name = 'J_day'
df_mu_star.index.name = 'J_day'
df_sigma.index.name = 'J_day'

# CSV 파일 저장 (행: 날짜, 열: 파라미터)
df_mu.to_csv(f'TS_morris_mu_{target_variable}.csv')
df_mu_star.to_csv(f'TS_morris_mu_star_{target_variable}.csv')
df_sigma.to_csv(f'TS_morris_sigma_{target_variable}.csv')

print(f">>> 저장 완료: 시계열 Mu, Mu_star, Sigma 파일 생성")

# ==============================================================================
# 5. 시계열 그래프 그리기 (Mu_star 기준 상위 5개 파라미터)
# ==============================================================================
print(">>> 시계열 그래프 생성 중...")

# 전체 기간 동안 평균적으로 Mu_star가 가장 높은 상위 5개 파라미터 추출
top_5_params = df_mu_star.mean().sort_values(ascending=False).head(5).index

plt.figure(figsize=(14, 6))

for param in top_5_params:
    plt.plot(df_mu_star.index, df_mu_star[param], label=param, alpha=0.8, linewidth=1.5)

plt.xlabel('Time (J_day)')
plt.ylabel('Mu* (Mean Absolute Influence)')
plt.title(f'Time-varying Sensitivity (Mu*) for Top 5 Parameters: {target_variable}')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

graph_filename = f'TS_morris_result_{target_variable}.png'
plt.savefig(graph_filename, dpi=300)
print(f">>> 그래프 저장 완료: {graph_filename}")
plt.show()