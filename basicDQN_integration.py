# 기본 DQN - noisynet, dueling, per 제거
import os
import numpy as np #"Numeric Python"의 약자, 대규모 다차원 배열과 행렬 연산에 필요한 다양한 함수와 메소드를 제공
import cupy as cp #NumPy 문법을 사용하며 NVIDIA GPU를 사용하여 행렬 연산으로 속도를 향상
import pandas as pd #파이썬 데이터 분석 라이브러리 
import tensorflow as tf
import torch #PyTorch의 핵심 모듈, 텐서연산과 자동미분 등 제공, GPU가속 지원원
import torch.nn as nn #신경망 레이어와 관련된 클래스 및 함수 포함
import torch.nn.functional as F #활성화 함수 등 포함
import torch.optim as optim #최적화 알고리즘 제공
import cv2
import random
import datetime
import math
import wandb
import socket
import threading

import torch.cuda as cuda #PyTorch에서 CUDA를 활용한 GPU 연산을 수행하는 모듈
import torch.backends.cudnn as cudnn #NVIDIA의 CuDNN (CUDA Deep Neural Network) 백엔드를 사용하여 CNN 연산을 최적화

import matplotlib.pyplot as plt #데이터 및 학습 과정을 시각화하는 라이브러리
from skimage.transform import resize # 이미지를 특정 크기로 리사이징 (강화학습 환경에서 입력 크기를 맞추는 데 사용)
from skimage.color import rgb2gray # RGB 이미지를 그레이스케일 변환 
from collections import deque # 빠른 큐(Queue) 연산을 위한 자료구조

import gc #Python의 가비지 컬렉터 (메모리 관리 용도)

from UAV_env import UAV_env #사용자 정의 환경 \ 자체 가상 환경
from Nstep_Buffer import n_step_buffer # N-step 경험 재생 버퍼

global_step = 0

# 차량 검출
# 차량 예측을 위한 CNN 모델 및 Feature Map 로드
vehicle_name = "RedCar"  # 예: "RedCar", "PurpleSportsCar" 등
model_path = rf'D:/cnn_Integration/CNN_detection/{vehicle_name}AttentionModel.h5'
feature_map_path = rf'D:/cnn_Integration/CNN_detection/{vehicle_name}_feature_map.npy'

# ========================== 차량 좌표 실시간 수신 코드 ==========================
import socket
import threading

vehicle_position = None  # 최신 차량 좌표 (CNN Ground Truth 용도)

def start_vehicle_socket_server(host='127.0.0.1', port=9999):
    def handle_client(client_socket):
        global vehicle_position
        while True:
            try:
                data = client_socket.recv(1024).decode()
                if not data:
                    break
                parts = data.strip().split(',')
                x = float(parts[0])
                z = float(parts[2])  # y는 생략
                vehicle_position = (x / 4, z / 4)  # CNN 스케일에 맞춤
            except Exception as e:
                print(f"좌표 수신 오류: {e}")
                break
        client_socket.close()

    def server_thread():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen(1)
        print(f"[소켓 서버 시작] {host}:{port}")
        while True:
            client_socket, addr = server.accept()
            print(f"[연결됨] {addr}")
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.start()

    threading.Thread(target=server_thread, daemon=True).start()

# 서버 실행
start_vehicle_socket_server()
###########################################################################################################
# 모델 및 특징맵 로드
try:
    model = tf.keras.models.load_model(model_path)
    if not os.path.exists(feature_map_path):
        raise FileNotFoundError(f"Feature map not found: {feature_map_path}")
    vehicle_feature_map = np.load(feature_map_path).reshape(1, -1)  # (256,) 형태로 로드 후 재배열
    #print(f"Loaded model and feature map for {vehicle_name}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# 이미지 전처리 함수
IMG_SIZE = (256, 256)

def tensor_to_image(tensor):
    # 텐서를 NumPy 배열로 변환하고, 채널 순서를 변경
    image = tensor.cpu().numpy().squeeze()  # 배치 차원 제거
    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV에서 사용 가능한 형식으로 변환
    return image

def load_test_image(image_tensor):
    image = tensor_to_image(image_tensor)
    image = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))  # 이미지 크기 조정
    image = np.expand_dims(image, axis=0).astype(np.float32)  # 배치 차원 추가
    return image


#예측된 좌표를 DQN의 입력으로 통합하는 함수
def cnn_predict_coords(obs):
    # CNN 모델을 통해 차량의 좌표 예측
    global global_step, vehicle_position
    test_image = load_test_image(obs)  # 여기서 이미지 경로를 받아올 수 있어야 합니다
    if test_image is not None:
        prediction = model.predict([test_image, vehicle_feature_map])[0]
        scaled_prediction = prediction / 4  # 64x64 환경으로 스케일링
        
        print(f"[{global_step}] Pre_vehicle_position: {scaled_prediction}")
        print(f"[{global_step}] vehicle_position: {vehicle_position}")
        
        global_step += 1
        return scaled_prediction
    return None

##########################################################################################################################    


#DQN
 
load = False # 모델을 불러올지 여부 False의 경우 새로운 모델 학습
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S") # 현재 날짜와 시간을 문자열로 변환
save_path = f"./best_model/{date_time}.pkl" # 모델 저장 경로
load_path = f"./best_model/obstacle.pkl" # 불러올 모델 경로로

project_exe = "../build/build_final/project_250306_Fog_integration.exe" #실행 프로젝트 파일

# wandb 에서 실험 추적 및 시각화 
wandb.init(
    project="Tracking_uav",
    entity="mnl431",
    config={
        "architecture": "DQN",
    }
)


class Q_network(nn.Module):
    def __init__(self, num_actions):
        super(Q_network, self).__init__()
        self.num_actions = num_actions
        self.image_cnn = nn.Sequential(
            # nn.Conv2d(입력채널 수, 출력 채널수, kernel_size=필터 크기, stride=필터 이동간격)
            # 64x64x4 -> 30x30x32
            nn.Conv2d(4, 32, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 30x30x32  -> 13x13x64
            nn.Conv2d(32, 64, kernel_size=6, stride=2, groups=1, bias=True),
            nn.GELU(),
            # 13x13x64 -> 10x10x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 10x10x64 -> 7x7x64
            nn.Conv2d(64, 64, kernel_size=4, stride=1, groups=1, bias=True),
            nn.GELU(),
            # 7x7x64 -> 5x5x64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, groups=1, bias=True)
            # 1600
        )

        self.ray_fc = nn.Sequential(
            nn.Linear(30, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 32)
            # 32
        )

        self.fc_connected = nn.Sequential(
            nn.Linear(1632, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_actions)
        )

        self.init_weights(self.image_cnn)
        self.init_weights(self.ray_fc)
        self.init_weights(self.fc_connected)

    # 가중치 초기화
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Q-value 출력 (CNN+FCLayer) 
    #UAV 카메라이미지 + UAV ray 센서(9) + info 추가 환경 정보(21)
    def forward(self, camera, ray, info):
        batch = camera.size(0)  # 0의 크기를 반환하기 때문에 batch는 1이 됨
        image_fcinput = self.image_cnn(camera).view(batch, -1) #이미지 cnn 처리
       
        combined_input = torch.cat([ray, info], dim=2) 
        #print(combined_input.shape) # torch.Size([1, 1, 32])
        #print("Combined input values:", combined_input)
        #print("info input values:", info)
        ray_fcinput = self.ray_fc(combined_input).view(batch, -1)
        
        x = torch.cat([image_fcinput, ray_fcinput], dim=1)
        Q_values = self.fc_connected(x)

        return Q_values


class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU사용/없으면 CPU사용
        self.device = device 

        self.learning_rate = 0.00002  # 0.00002
        self.batch_size = 32 # 학습할 때 한번에 샘플링할 데이터 개수
        self.gamma = 0.95  # 0.95 보상에서 현재 가치 -> 95% 반영
        self.n_step = 1  # 2
        self.num_actions =  15

        self.epsilon = 1 # 초기에 100% 확률로 랜덤 선택 
        self.initial_epsilon = 1.0
        self.epsilon_decay_rate = 0.8 # 
        self.final_epsilon = 0.1  # 최종 값
        self.epsilon_decay_period = 1000000  # 100000 #(231126) 감쇠가 적용되는 총 학습 단계
        self.epsilon_cnt = 0 
        self.epsilon_max_cnt = 1

        # self.epsilon_decay = 0.000006 #0.000006
        self.soft_update_rate = 0.005  # 0.01 타겟 네트워크를 조금씩 업데이트하여 학습안정성
        self.rate_update_frequency = 150000 #몇번의 학습 후 업데이트 할지 결정
        self.max_rate = 0.04 # 최대 업데이트 비율 제한

        self.data_buffer = deque(maxlen=15000) # 최근 15000개의 데이터 저장하고 랜덤 샘플링
        #self.nstep_memory = n_step_buffer(n_step=self.n_step)

        self.action_history = deque([0, 0, 0, 0], maxlen=4) # 최근 4개의 행동을 저장

        self.policy_net = Q_network(self.num_actions).to(self.device) # 현재 상태에서 최적 행동 예측
        self.Q_target_net = Q_network(self.num_actions).to(self.device) # 일정 주기마다 업데이트
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate) # adam 옵티마이저
        self.Q_target_net.load_state_dict(self.policy_net.state_dict()) # 초기에는 policy_net과 target_net을 동일하게 설정 
        self.Q_target_net.eval() # target은 학습되지 않음

        self.epi_loss = 0

        # self.data_buffer = deque(maxlen=20000)

        if load == True:
            print("Load trained model..")
            self.load_model()

    def update_epsilon(self, current_step):
        # if 999000 <= current_step <= 1001000:
        #     logging.info(
        #         f"Current Step: {current_step}, Epsilon: {self.epsilon}, Initial Epsilon: {self.initial_epsilon}, Epsilon Count: {self.epsilon_cnt}")

        if self.epsilon_cnt == self.epsilon_max_cnt:
            pass
            # self.epsilon = self.final_epsilon
        else:
            if current_step % self.epsilon_decay_period == 0:
                self.epsilon_cnt += 1
                # self.initial_epsilon = self.initial_epsilon * self.epsilon_decay_rate
                # self.epsilon = max(self.initial_epsilon, self.final_epsilon)
                self.epsilon = self.final_epsilon
            else:
                cos_decay = 0.5 * (1 + math.cos(
                    math.pi * (current_step % self.epsilon_decay_period) / self.epsilon_decay_period))
                self.epsilon = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * cos_decay

    def epsilon_greedy(self, Q_values):
        # 난수 생성
        if np.random.random() < self.epsilon:
            # action을 random하게 선택
            action = random.randrange(self.num_actions)
            return action
        else:
            # 학습된 Q value 값중 가장 큰 action 선택
            return Q_values.argmax().item()

    # model 저장
    def save_model(self):
        torch.save({
            'state': self.policy_net.state_dict(),
            'optim': self.optimizer.state_dict()},
            save_path)
        return None

    # model 불러오기
    def load_model(self):
        checkpoint = torch.load(load_path)
        self.policy_net.load_state_dict(checkpoint['state'])
        self.Q_target_net.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        return None

    def store_trajectory(self, traj):
        self.data_buffer.append(traj)

    # 1. resizing : 64 * 64, gray scale로
    def re_scale_frame(self, obs):
        obs = cp.array(obs)
        obs = cp.asnumpy(obs)
        obs = np.transpose(obs, (1, 2, 0))
        obs = resize(rgb2gray(obs), (64, 64))
        return obs

    # 2. image 4개씩 쌓기
    def init_image_obs(self, obs):
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis=0)
        frame_obs = cp.array(frame_obs)  # cupy 배열로 변환
        return frame_obs

    # 3. 4장 쌓인 Image return
    def init_obs(self, obs):
        return self.init_image_obs(obs)

    def camera_obs(self, obs):
        camera_obs = cp.array(obs)  # cupy 배열로 변환
        # print(obs.shape) # 4 64 64 3
        camera_obs = cp.expand_dims(camera_obs, axis=0)
        camera_obs = torch.from_numpy(cp.asnumpy(camera_obs)).to(self.device)  # GPU로 전송
        return camera_obs

    def ray_obs(self, obs):
        ray_obs = cp.array(obs)  # cupy 배열로 변환
        ray_obs = cp.expand_dims(ray_obs, axis=0)
        ray_obs = torch.from_numpy(cp.asnumpy(ray_obs)).unsqueeze(0).to(self.device)  # GPU로 전송
        return ray_obs

    def ray_obs_cpu(self, obs):
        obs_gpu = cp.asarray(obs)
        obs_gpu = cp.reshape(obs_gpu, (1, -1))
        return cp.asnumpy(obs_gpu)

    # FIFO, 4개씩 쌓기

    def accumulated_image_obs(self, obs, new_frame):
        temp_obs = obs[1:, :, :]  # 4x3x64x64에서 제일 오래된 이미지 제거 => 3x3x64x64
        new_frame = self.re_scale_frame(new_frame)  # 3x64x64
        # plt.imshow(new_frame)
        # plt.show()
        temp_obs = cp.array(temp_obs)  # cupy 배열로 변환
        new_frame = cp.array(new_frame)  # cupy 배열로 변환
        new_frame = cp.expand_dims(new_frame, axis=0)  # 1x3x64x64
        frame_obs = cp.concatenate((temp_obs, new_frame), axis=0)  # 4x3x64x64
        frame_obs = cp.asnumpy(frame_obs)  # 다시 numpy 배열로 변환
        return frame_obs

    def accumlated_all_obs(self, obs, next_obs):
        return self.accumulated_image_obs(obs, next_obs)

    def update_action_history(self, action):
        self.action_history.append(action)
        return list(self.action_history)

    # action 선택, discrete action 15개 존재
    # obs shape : torch.Size([1, 4, 64, 64])
    def train_policy(self, obs_camera, obs_ray, info_data):
        predicted_coords = cnn_predict_coords(obs_camera)
        
       # predicted_coords를 numpy 배열에서 PyTorch 텐서로 변환
        if isinstance(predicted_coords, np.ndarray):
            predicted_coords = torch.from_numpy(predicted_coords).float().to(obs_camera.device)

        # predicted_coords의 차원 확장
        predicted_coords = predicted_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, 2) 크기로 확장

        # info 텐서의 마지막 두 자리에 예측된 좌표를 삽입
        info_data[..., -2:] = predicted_coords

        Q_values = self.policy_net(obs_camera, obs_ray, info_data)
        max_q = Q_values.max()
        action = self.epsilon_greedy(Q_values)

        return action, Q_values[0][action], max_q
        #return action, Q_values[0][action], max_q

    def batch_torch_obs(self, obs):
        obs = [cp.asarray(ob) for ob in obs]  # obs의 모든 요소를 cupy 배열로 변환
        obs = cp.stack(obs, axis=0)  # obs를 축 0을 기준으로 스택
        obs = cp.squeeze(obs, axis=0) if obs.shape[0] == 1 else obs  # 첫 번째 축 제거
        obs = cp.asnumpy(obs)  # 다시 numpy 배열로 변환
        obs = torch.from_numpy(obs).to(self.device)  # torch tensor로 변환
        return obs

    def batch_ray_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    def batch_info_obs(self, obs):
        obs = cp.asarray(obs)  # cupy 배열로 변환
        # obs = cp.expand_dims(obs, axis=0)  # 새로운 축 추가
        obs = torch.from_numpy(cp.asnumpy(obs)).to(self.device)  # torch tensor로 변환
        return obs

    # update target network
    # Q-Network의 파라미터를 target network 복사
    def update_target(self, step):
        if step % self.rate_update_frequency == 0:
            self.soft_update_rate += 0.001

        self.soft_update_rate = min(self.soft_update_rate, self.max_rate)
        # print("soft_rate: ", self.soft_update_rate)

        policy_dict = self.policy_net.state_dict()
        target_dict = self.Q_target_net.state_dict()

        # 소프트 업데이트 수행
        for name in target_dict:
            target_dict[name] = (1.0 - self.soft_update_rate) * target_dict[name] + self.soft_update_rate * policy_dict[
                name]

        # 업데이트된 가중치를 타겟 네트워크에 설정
        self.Q_target_net.load_state_dict(target_dict)
        # self.Q_target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, step, update_target):

        # mini_batch, idxs, IS_weights = self.memory.sample(self.batch_size)
        random_mini_batch = random.sample(self.data_buffer, self.batch_size)
        # #epsilon decaying
        # self.epsilon -= self.epsilon_decay
        # #min of epsilon : 0.05
        # self.epsilon = max(self.epsilon, 0.1) # 약 200000step 이후 최솟값
        # #print("epsilon: ", self.epsilon)

        self.obs_camera_list, self.obs_ray_list, self.info_list, self.action_list, self.reward_list, self.next_obs_camera_list, self.next_obs_ray_list, self.next_info_list, self.mask_list = zip(
            *random_mini_batch)

        # tensor
        obses_camera = self.batch_torch_obs(self.obs_camera_list)
        obses_ray = self.batch_ray_obs(self.obs_ray_list)
        # print("camera:",obses_camera.shape)
        # print("ray:",obses_ray.shape)

        actions = torch.LongTensor(self.action_list).unsqueeze(1).to(self.device)

        rewards = torch.Tensor(self.reward_list).to(self.device)
        next_obses_camera = self.batch_torch_obs(self.next_obs_camera_list)
        next_obses_ray = self.batch_ray_obs(self.next_obs_ray_list)

        masks = torch.Tensor(self.mask_list).to(self.device)

        obs_info = self.batch_info_obs(self.info_list)
        next_obs_info = self.batch_info_obs(self.next_info_list)
        # print("info:",obs_info.shape)

        Q_values = self.policy_net(obses_camera, obses_ray, obs_info)
        q_value = Q_values.gather(1, actions).view(-1)
        # print(q_value)

        # get target, y(타겟값) 구하기 위한 다음 state에서의 max Q value
        # target network에서 next state에서의 max Q value -> 상수값
        with torch.no_grad():
            target_q_value = self.Q_target_net(next_obses_camera, next_obses_ray, next_obs_info).max(1)[0]

        Y = (rewards + masks * (self.gamma ** self.n_step) * target_q_value).clone().detach()

        MSE = nn.MSELoss()
        #           input,  target
        loss = MSE(q_value, Y.detach())
        # errors = F.mse_loss(q_value, Y, reduction='none')

        # 우선순위 업데이트
        # for i in range(self.batch_size):
        #     tree_idx = idxs[i]
        #     self.memory.batch_update(tree_idx, errors[i])

        self.optimizer.zero_grad()

        # loss 정의 (importance sampling)
        # loss =  (torch.FloatTensor(IS_weights).to(self.device) * errors).mean()
        # 10,000번의 episode동안 몇 번의 target network update가 있는지
        # target network update 마다 max Q-value / loss function 분포

        # # tensor -> list
        # # max Q-value 분포
        # tensor_to_list_q_value = target_q_value.tolist()
        # # max_Q 값들(batch size : 32개)의 평균 값
        # list_q_value_avg = sum(tensor_to_list_q_value)/len(tensor_to_list_q_value)
        # self.y_max_Q_avg.append(list_q_value_avg)

        # # loss 평균 분포(reduction = mean)
        # loss_in_list = loss.tolist()
        # self.y_loss.append(loss_in_list)

        # backward 시작

        loss.backward()
        self.optimizer.step()

        self.epi_loss += loss.item()
        # --------------------------------------------------------------------


def main():
    env = UAV_env(time_scale=2.0, filename=project_exe, port=11300)

    cudnn.enabled = True
    cudnn.benchmark = True

    score = 0
    # episode당 step
    episode_step = 0
    # 전체 누적 step
    step = 0
    update_target = 1000  # 2000
    initial_exploration = 10000  # 10000

    agent = Agent()  # 에이전트 인스턴스

    if load:
        agent.load_model()

    for epi in range(5001):
        obs = env.reset()

        obs_camera = torch.Tensor(obs[0]).squeeze(dim=0)
        # (84, 84, 3) -> (64, 64, 1) -> 4장씩 쌓아 (64, 64, 4)
        # 같은 Image 4장 쌓기 -> 이후 action에 따라 환경이 바뀌고, 다른 Image data 쌓임
        obs_camera = agent.init_obs(obs_camera)

        obs_height = obs[1]
        obs_ray = obs[2]
        # # c#에서 받아온 obs
        # #[0.         1.         1.         0.         1.         1.
        # # 0.         1.         1.         0.         1.         1.
        # # 1.         0.         0.17635795 0.         1.         1.
        # # 0.         1.         1.         0.         1.         1.
        # # 0.         1.         1.]
        #
        
        idx_list = [2, 5, 8, 11, 14, 17, 20, 23]
        obs_ray_tensor = [obs_ray[i] for i in range(27) if i in idx_list]
        obs_ray_tensor = np.append(obs_ray_tensor, obs_height[2])
        obs_ray_tensor = torch.Tensor(obs_ray_tensor)
        obs[3] = np.concatenate((obs[3], [0, 0, 0, 0]))  # action 4step 추가
        obs_info = torch.Tensor(obs[3])

        while True:

            # action 선택
            dis_action, estimate_Q, max_est_Q = agent.train_policy(agent.camera_obs(obs_camera),
                                                                   agent.ray_obs(obs_ray_tensor),
                                                                   agent.ray_obs(obs_info))

            if episode_step == 0:
                print("Max Q-value: ", max_est_Q.cpu().item())
                print("Epsilon:", agent.epsilon)

            # action에 따른 step()
            # next step, reward, done 여부
            next_obs, reward, done = env.step(dis_action)

            # state는 camera sensor로 얻은 Image만
            next_obs_camera = next_obs[0]
            next_obs_height = next_obs[1]
            next_obs_ray = next_obs[2]
            next_obs_info = next_obs[3]

            # todo next obs ray
            next_obs_ray_tensor = [next_obs_ray[i] for i in range(27) if i in idx_list]
            next_obs_ray_tensor = np.append(next_obs_ray_tensor, next_obs_height[2])
            next_obs_ray_tensor = torch.Tensor(next_obs_ray_tensor)

            next_obs_camera = torch.Tensor(next_obs_camera).squeeze(dim=0)
            # step이 증가함에 따라 4장 중 1장씩 밀기(FIFO)
            next_obs_camera = agent.accumlated_all_obs(obs_camera, next_obs_camera)
            next_obs_info = np.concatenate((next_obs_info, agent.update_action_history(dis_action)))
            next_obs_info = torch.Tensor(next_obs_info)

            #next_obs_info = np.concatenate((next_obs_info, agent.update_action_history(dis_action)))



            mask = 0 if done else 1
            # print("%d번째 step에서의 reward : %f, action speed : %f"%(step, reward, action_speed))
            score += reward

            agent.store_trajectory(
                [obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_info), dis_action, reward,
                 next_obs_camera, agent.ray_obs_cpu(next_obs_ray_tensor), agent.ray_obs_cpu(next_obs_info), mask])
            # return_trajectory = agent.nstep_memory.append(cur_sample) # 멀티 스텝 학습을 위해 리턴값과 다음 상태 값 반환

            # if return_trajectory is not None:
            #     n_step_rewards, next_cam, next_ray, next_sig, last_mask = return_trajectory
            #     # 샘플 수정
            #     return_sample = (obs_camera, agent.ray_obs_cpu(obs_ray_tensor), agent.ray_obs_cpu(obs_info), dis_action,
            #                      n_step_rewards, next_cam, agent.ray_obs_cpu(next_ray), agent.ray_obs_cpu(next_sig), last_mask)

            #     # TD-error를 위한 max Target Q-value 계산
            #     with torch.no_grad():
            #         target_Q = agent.Q_target_net(agent.camera_obs(next_cam), agent.ray_obs(next_ray), agent.ray_obs(next_sig)).max(1)[0]
            #         target_value = torch.tensor((n_step_rewards + last_mask * (agent.gamma ** agent.n_step)* target_Q).item()).to("cuda:0")

            #     # print("e:",  estimate_Q) # --> 스칼라 텐서
            #     # print("t: ", torch.tensor(target_value)) #--> 스칼라

            #     # 우선 순위 계산을 위한 TD-error 계산
            #     td_error = F.mse_loss(estimate_Q,  target_value.detach())

            #     # 우선 순위 리플레이버퍼에 저장
            #     agent.memory.store(td_error, return_sample)

            obs_camera = next_obs_camera
            obs_ray_tensor = next_obs_ray_tensor
            obs_info = next_obs_info
            #print(obs_ray_tensor.shape)

            # SumTree 노드 수가 배치 사이즈 이상 되면 학습
            if step > agent.batch_size:
                # if agent.memory.tree.n_entries > agent.n_step:
                agent.train(step, update_target)

                # 모델 저장
                if step % 2000 == 0:
                    agent.save_model()

                # 타겟 네트워크 업데이트
                if step % update_target == 0:
                    agent.update_target(step)

            episode_step += 1
            step += 1
            agent.update_epsilon(step)

            if done:
                cuda.empty_cache()
                gc.collect()
                break

        print('%d 번째 episode의 총 step: %d' % (epi + 1, episode_step))
        print('True_score: %f' % score)
        print('Total step: %d\n' % step)


        # todo wandb
        wandb.log({
            "episode_step": episode_step,
            "score": score,
            "Average score": score / episode_step,
            "init Max Q": max_est_Q.cpu().item(),
            "Average Loss": agent.epi_loss / episode_step if episode_step != 0 else 0,
            "Epsilon": agent.epsilon,
            "Episode": epi},
            step=epi)

        agent.epi_loss = 0
        score = 0
        episode_step = 0


if __name__ == '__main__':
    main()
