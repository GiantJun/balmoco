from os.path import join, exists, basename, dirname
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

mean_RGB = 0.36541346
std_RGB = 0.1908806


class Bal_DLDataset(Dataset):
    """自定义 Dataset 类"""
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        data_frame = pd.read_csv(csv_path)
        self.class_num=2
        self.class_names=['non-BA','BA']

        self.data = data_frame['Image_path:'].to_list()
        self.targets = data_frame['Label:'].to_list()
    
    def get_XY(self):
        return self.data, self.targets

    def __len__(self):
        """返回 dataset 大小"""
        return len(self.data)

    def set_transforms(self, tf):
        self.transform = tf

    def __getitem__(self, idx):
        """根据 idx 从 图片名字-类别 列表中获取相应的 image 和 label"""     
        if not exists(self.data[idx]):
            data_dir = dirname(self.data[idx])
            name, suffix = basename(self.data[idx]).split('.')
            img_path = join(data_dir, name+'.'+suffix.lower())
        else:
            img_path = self.data[idx]
        image = Image.open(img_path)
        target = self.targets[idx]

        return self.transform(image), target

# if __name__ == '__main__':
#     # 计算灰度图时使用
#     tf = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     select_list=list(range(0,3))
#     dataset = DLDataset('/media/huangyujun/disk/data/深度学习前沿大作业_超声数据集/内部数据集/Original_train_dataset/label.csv',transform=tf)
#     dataloader = DataLoader(dataset=dataset, batch_size=int(len(dataset)), shuffle=False, num_workers=4)
#     img_l = []
#     for imgs, label, img_dir in dataloader:
#         imgs = imgs.numpy()
#         print(imgs.shape)
#         for i in range(len(select_list)):
#             # 对于单通道时计算mean和std使用
#             img_l.append(imgs[:,i,:,:])
#         all = np.vstack(img_l)
#         print(all.shape)
#         mean = np.mean(all)
#         std = np.std(all)
#     print(mean, std)

