import os
import numpy as np
from numpy import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import cv2
import IPython.display as ipd
import librosa
import librosa.display
from keras.utils import to_categorical

labels = ['up', 'down', 'start', 'stop', 'turn left', 'turn right', 'slow down']
label_encoder = {label:i for i, label in enumerate(labels)}
print(label_encoder['down'])
print(label_encoder['slow down'])

x_test = []
y_test = []

test_files = '/content/drive/MyDrive/ASR_multi_channel/multi channel3/test'

for label in labels:
  test_label_path = os.path.join(test_files, label)

  for file in os.listdir(test_label_path):
    audio, sr = librosa.load(os.path.join(test_label_path, file), sr = 22050)
    s1 = librosa.feature.melspectrogram(y=audio, sr = 22050)
    s2 = librosa.power_to_db(s1, ref=np.max)
    x_test.append(s2)
    y_test.append(labels.index(label))

x_train = []
y_train = []

phone_files = "/content/drive/MyDrive/ASR_multi_channel/multi channel3/phone_record"
lap_files = "/content/drive/MyDrive/ASR_multi_channel/multi channel3/lap_record"

for label in labels:
    phone_label_path = os.path.join(phone_files, label)
    lap_label_path = os.path.join(lap_files, label)

    for phone_file in os.listdir(phone_label_path):
        audio, sr = librosa.load(os.path.join(phone_label_path, phone_file), sr=22050)
        s1 = librosa.feature.melspectrogram(y=audio, sr=22050)
        s2 = librosa.power_to_db(s1, ref=np.max)
        x_train.append(s2)
        y_train.append(labels.index(label))

    for lap_file in os.listdir(lap_label_path):
        audio, sr = librosa.load(os.path.join(lap_label_path, lap_file), sr=22050)
        s1 = librosa.feature.melspectrogram(y=audio, sr=22050)
        s2 = librosa.power_to_db(s1, ref=np.max)
        x_train.append(s2)
        y_train.append(labels.index(label))

max_height = max([x.shape[0] for x in x_train])
max_width = max([x.shape[1] for x in x_train])

x_train_padded = []
for spectrogram in x_train:
  height, width = spectrogram.shape
  padded_spectrogram = np.pad(spectrogram, ((0, max_height - height), (0, max_width - width)), mode = 'constant', constant_values = 0)
  x_train_padded.append(padded_spectrogram)

x_test_padded = []
for spectrogram in x_test:
  height, width = spectrogram.shape
  padded_spectrogram = np.pad(spectrogram, ((0, max_height - height), (0, max_width - width)), mode = 'constant', constant_values = 0)
  x_test_padded.append(padded_spectrogram)

X_train = np.array(x_train_padded)
Y_train = np.array(y_train)
X_test = np.array(x_test_padded)
Y_test = np.array(y_test)

from sklearn.model_selection import train_test_split
x_train_final, x_valid_final, y_train_final, y_valid_final = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

num_classes = 7
y_train_final = to_categorical(y_train_final, num_classes)
y_valid_final = to_categorical(y_valid_final, num_classes)
y_test_final = to_categorical(y_test, num_classes)

class CustomEnv:
  def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
    self.model = self.modelCNN(0.001)
    self.weight_file = 'multi_weights.h5'
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid
    self.x_test = x_test
    self.y_test = y_test

    self.learning_rate = 0.001
    self.observation_space = [0.001, 0.6499999761581421]
    self.action_space = [0, 1]
    self.history = None
    self.rewards = []

  def modelCNN(self, lr):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 303, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate = lr), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.load_weights('multi_weights.h5')
    return model

  def step(self, action):
    learning_rate = self.learning_rate
    reward = 0
    if action == 0:
      learning_rate *= 1 + random.rand()
    elif action == 1:
      learning_rate *= random.rand()
#    else:
#      learning_rate = self.learning_rate

    self.model = self.modelCNN(learning_rate)
    self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=32, epochs=20)
    test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)

    if test_acc > self.observation_space[1]:
      reward += (test_acc - self.observation_space[1])*2
    elif test_acc < self.observation_space[1]:
      reward -= (test_acc - self.observation_space[1])*4
    else:
      reward += 0

    return test_loss, test_acc, reward, learning_rate


  def update(self, action):
    test_loss, test_acc, reward, learning_rate = self.step(action)
    self.observation_space = [learning_rate, test_acc]
    self.rewards.append(reward)

  def reset(self):
    self.observation_space = [0.001, 0.6499999761581421]
    self.rewards = []
    self.learning_rate = 0.001
    self.model = self.modelCNN(0.001)

class CustomEnv2:
  def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
    self.model = self.modelCNN(0.001)
    self.weight_file = 'multi_weights.h5'
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid
    self.x_test = x_test
    self.y_test = y_test

    self.learning_rate = 0.001
    self.observation_space = [0.001, 0.5571428537368774]
    self.action_space = [0, 1]
    self.history = None
    self.rewards = []

  def modelCNN(self, lr):
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (128, 303, 1)))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(32, (3, 3), activation = 'relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(64, (3, 3), activation = 'relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Flatten())
    model2.add(Dense(64, activation = 'relu'))
    model2.add(Dense(7, activation = 'softmax'))

    model2.compile(optimizer=Adam(learning_rate=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model2.load_weights('model2.h5')
    return model2

  def step(self, action):
    learning_rate = self.learning_rate
    reward = 0
    if action == 0:
      learning_rate *= 1 + random.rand()
    elif action == 1:
      learning_rate *= random.rand()
#    else:
#      learning_rate = self.learning_rate

    self.model = self.modelCNN(learning_rate)
    self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=32, epochs=20)
    test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)

    if test_acc > self.observation_space[1]:
      reward += (test_acc - self.observation_space[1])*2
    elif test_acc < self.observation_space[1]:
      reward -= (test_acc - self.observation_space[1])*4
    else:
      reward += 0

    return test_loss, test_acc, reward, learning_rate


  def update(self, action):
    test_loss, test_acc, reward, learning_rate = self.step(action)
    self.observation_space = [learning_rate, test_acc]
    self.rewards.append(reward)

  def reset(self):
    self.observation_space = [0.001, 0.5571428537368774]
    self.rewards = []
    self.learning_rate = 0.001
    self.model = self.modelCNN(0.001)

class CustomEnv3:
  def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
    self.model = self.modelCNN(0.001)
    self.weight_file = 'multi_weights.h5'
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid
    self.x_test = x_test
    self.y_test = y_test

    self.learning_rate = 0.001
    self.observation_space = [0.001, 0.5428571701049805]
    self.action_space = [0, 1]
    self.history = None
    self.rewards = []

  def modelCNN(self, lr):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 303, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(7, activation='softmax'))


    model.compile(optimizer=Adam(learning_rate = lr), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.load_weights('model3.h5')
    return model

  def step(self, action):
    learning_rate = self.learning_rate
    reward = 0
    if action == 0:
      learning_rate *= 1 + random.rand()
    elif action == 1:
      learning_rate *= random.rand()
#    else:
#      learning_rate = self.learning_rate

    self.model = self.modelCNN(learning_rate)
    self.history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid), batch_size=32, epochs=20)
    test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)

    if test_acc > self.observation_space[1]:
      reward += (test_acc - self.observation_space[1])*2
    elif test_acc < self.observation_space[1]:
      reward -= (test_acc - self.observation_space[1])*4
    else:
      reward += 0

    return test_loss, test_acc, reward, learning_rate


  def update(self, action):
    test_loss, test_acc, reward, learning_rate = self.step(action)
    self.observation_space = [learning_rate, test_acc]
    self.rewards.append(reward)

  def reset(self):
    self.observation_space = [0.001, 0.5428571701049805]
    self.rewards = []
    self.learning_rate = 0.001
    self.model = self.modelCNN(0.001)

class DQN:
  def __init__(self, env):
    self.env = env
    self.state_size = 1
    self.state = env.observation_space[1]
    self.action_space = env.action_space
    self.action_size = 2
    self.batch_size = 32
    self.gamma = 0.99
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.replay_memory = []
    self.max_memory_size = 1000
    self.model = self.build_model()
    self.total_lr = []
    self.total_state = []
    self.total_loss = []
    self.total_score = []

  def build_model(self):
    state_size = self.state_size
    action_size = self.action_size
    model = Sequential()
    model.add(Flatten(input_shape = (1, state_size)))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(action_size, activation = 'linear'))
    model.compile(loss = 'mse', optimizer = Adam(learning_rate = 0.001))
    return model

  def train(self):
    env = self.env
    model = self.model
    for episode in range(100):
      self.env.reset()
      state = self.state
      score = 0

      # chọn action: quay random, nếu ra số <= epsilon thì action = random, else action = dự đoán của dqn
      if np.random.rand() <= self.epsilon:
        action = np.random.randint(self.action_size)
      else:
        action = np.argmax(model.predict(np.array([state])))

      # thực hiện step và update ở env
      self.env.update(action)
      next_state = self.env.observation_space[1]
      self.total_state.append(next_state)
      lr = self.env.observation_space[0]
      self.total_lr.append(lr)
      reward = self.env.rewards[-1]
      score += reward
      self.total_score.append(score)

      # thêm dữ liệu vào replay_memory
      self.replay_memory.append((state, action, reward, next_state))
      if len(self.replay_memory) > self.max_memory_size:
        self.replay_memory.pop(0)
      state = next_state

      # training model dqn
      if len(self.replay_memory) >= self.batch_size:
        # lấy số thứ tự random
        samples = np.random.choice(len(self.replay_memory), self.batch_size)
        for i in samples:
          state, action, reward, next_state = self.replay_memory[i]
          target = reward
          if True:
            # state là testing accuracy thì hợp lý hơn
            target = reward + self.gamma * np.max(model.predict(np.array([next_state])))
          target_f = model.predict(np.array([state]))
          target_f[0, action] = target
        his = model.fit(np.array([state]), np.array([target_f]), epochs = 1, verbose = 0)
        loss = his.history['loss']
        self.total_loss.append(loss)

      if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
      print(f"episode {episode + 1}, score: {score}")

#main:
env = CustomEnv(x_train_final, y_train_final, x_valid_final, y_valid_final, X_test, y_test_final)
agent = DQN(env)
agent.train()