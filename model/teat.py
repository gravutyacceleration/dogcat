import device
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MyDataset: 훈련용/테스트용 데이터를 처리하는 클래스
class MyDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.url = f'./{folder}'  # 'training_set' 또는 'test_set' 폴더로 변경
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.categorys = []
        self.Imgs = []
        self.len = 0

        for subfolder in ['dogs', 'cats']:
            subfolder_path = os.path.join(self.url, subfolder)  # 경로 변경
            print(f"Reading files from: {subfolder_path}")  # 경로 출력 확인
            if not os.path.exists(subfolder_path):  # 폴더가 존재하는지 확인
                print(f"폴더가 없습니다: {subfolder_path}")
                continue

            for file in os.listdir(subfolder_path):
                image = Image.open(os.path.join(subfolder_path, file)).convert('RGB')
                self.Imgs.append(self.transform(image))
                if subfolder == 'dog':
                    self.categorys.append(torch.FloatTensor([0]))
                else:
                    self.categorys.append(torch.FloatTensor([1]))

        self.len = len(self.categorys)
        print(f"Loaded {self.len} images from {self.url}")  # 데이터셋 길이 출력

    def __getitem__(self, index):
        return self.Imgs[index], self.categorys[index]

    def __len__(self):
        return self.len



# CNN 모델 정의
class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.fc_layer(out)
        return out


# 데이터셋 로딩
TrainDataset = MyDataset('training_set')  # 훈련 데이터셋
TestDataset = MyDataset('test_set')    # 테스트 데이터셋

Train_DataLoader = DataLoader(TrainDataset, batch_size=128, shuffle=True)
Test_DataLoader = DataLoader(TestDataset, batch_size=128)

# 모델 및 옵티마이저 설정
model = CNN_Model().to(device)  # 모델을 GPU로 이동
optimizer = optim.SGD(model.parameters(), lr=0.005)
nb_epochs = 10
history = []

# 학습 루프
for i in range(nb_epochs):
    model.train()  # 모델을 학습 모드로 설정
    for j, (image, label) in enumerate(Train_DataLoader):
        image, label = image.to(device), label.to(device)  # 데이터를 GPU로 이동

        output = model(image)
        loss = F.binary_cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = output >= 0.5  # 예측값이 0.5 이상이면 True
        correct_prediction = prediction.float() == label  # 예측과 실제값이 같은지 확인
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도 계산

        history.append(loss.item())
        print(f'Epoch {i+1}/{nb_epochs}, [{j+1}] Cost: {loss.item():.6f} Accuracy: {accuracy * 100:.2f}%')

# 테스트 루프
model.eval()  # 모델을 평가 모드로 설정
correct = 0
total = 0

with torch.no_grad():  # 평가 시에는 gradient 계산을 하지 않음
    for image, label in Test_DataLoader:
        image, label = image.to(device), label.to(device)  # 데이터를 GPU로 이동

        output = model(image)
        prediction = output >= 0.5  # 예측값이 0.5 이상이면 True
        correct_prediction = prediction.float() == label  # 예측과 실제값이 같은지 확인
        correct += correct_prediction.sum().item()
        total += len(label)

print(f'Acc : {correct / total:.4f}')
