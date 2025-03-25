from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.url = f'./{folder}'  # 'train' 또는 'test' 폴더
        print(f"Loading dataset from: {self.url}")  # 경로 출력 추가
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.categorys = []
        self.Imgs = []
        self.len = 0

        for subfolder in ['dogs', 'cats']:
            subfolder_path = os.path.join(self.url, subfolder)
            print(f"Reading files from: {subfolder_path}")  # 폴더 경로 출력
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
