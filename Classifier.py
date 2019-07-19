from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_tests.DistracedDriver import DistractedDriver
from pylibdl.optim import Adam
import pylibdl as libdl
from pylibdl.modules import *
np.random.seed(0)
import time


class MyResNet(Module):

    def __init__(self):
        super().__init__()
        self.initial = Sequential(
            Conv2D(3, 64, 5, stride=2, bias=False),  # 64x64
            BatchNorm2d(64),
            MaxPool2d(2),  # 32x32
            LeakyReLU()
        )

        self.res1 = Sequential(
            Conv2D(64, 64, 3, stride=1, bias=False),  # 16x16
            BatchNorm2d(64),
            LeakyReLU(),
            Conv2D(64, 64, 3, stride=2, bias=False),  # stride here
            BatchNorm2d(64)
        )
        self.adapt1 = Sequential(
            Conv2D(64, 64, 1, stride=2, bias=False),
            BatchNorm2d(64)
        )

        self.res2 = Sequential(
            Conv2D(64, 128, 3, stride=1, bias=False),
            BatchNorm2d(128),
            LeakyReLU(),
            Conv2D(128, 128, 3, stride=1, bias=False),
            BatchNorm2d(128)
        )
        self.adapt2 = Sequential(
            Conv2D(64, 128, 1, stride=1, bias=False),
            BatchNorm2d(128)
        )

        self.l2 = Linear(16*16*128, 10)

    def forward(self, x):
        x = self.initial(x)
        x = libdl.leaky_relu(self.res1(x) + self.adapt1(x))
        x = libdl.leaky_relu(self.res2(x) + self.adapt2(x))
        x = libdl.reshape(x, (16*16*128, -1))
        return self.l2(x)


if __name__ == '__main__':

    def validate():
        accs = []
        for imgs, labels in val_loader:
            imgs = tensor(imgs.numpy().transpose(1, 2, 3, 0))
            pred = model(imgs).numpy()
            accs.append((pred.argmax(0) == labels.numpy()).mean())
        return np.mean(accs)


    path = '/home/superbabes/Downloads/ddriver'
    lr = 1e-4
    batch_size = 64

    transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])

    train_set = DistractedDriver(path, val=False, transforms=transforms)
    val_set = DistractedDriver(path, val=True, transforms=transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, drop_last=False)

    model = MyResNet()

    optim = Adam(model.parameter(), lr)

    last = time.time()
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = tensor(imgs.numpy().transpose(1, 2, 3, 0))
        labels = tensor(np.eye(10)[labels.numpy()].transpose())

        pred = model(imgs)
        loss = libdl.cross_entropy_with_logits(pred, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(i, loss.numpy())
        if i == 150:
            break
        if i % 25 == 0 and i > 0:
            print(i, validate())
    model.save('model')
    print(validate())
    print(time.time() - last)

