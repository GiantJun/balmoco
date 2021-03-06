from torch.utils.data import Dataset
from os.path import join, splitext
from os import listdir
from torch import cat
from PIL import Image

class UltrasoundDataset(Dataset):
    """自定义 Dataset 类"""
    
    def __init__(self, root, transform=None, select_list=list(range(0,9)),dataset_type='train'):
        """
        输入：root_dir：数据子集的根目录
        dataset_type如果为“test”则会获取测试集数据
        说明：此处的root_dir 需满足如下的格式：
        root_dir/
        |-- class0
        |   |--image_group0
        |   |   |--1.png
        |   |   |--2.png
        |   |   |--3.png
        |   |--image_group2
        |   |   |--1.png
        |   |   |--2.png
        |-- class1
        """
        self.transform = transform
        self.item_list = []
        self.select_list = select_list  # 选择9幅图中的几幅（从0开始计）
        self.class_names = None
        self.class_num = None
        if dataset_type == 'all':
            root_dir = join(root,'train')
            # 训练集测试集下，各类别名默认为一致
            self.class_names = listdir(root_dir)
            self.class_names.sort()
            self.class_num = len(self.class_names)
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                for type in ['train', 'valid', 'test']:
                    root_dir = join(root,type)
                    # 训练集测试集下，各类别名默认为一致
                    class_dir = join(root_dir, item)
                    img_dirs = listdir(class_dir)
                    # img_dirs.sort()
                    self.item_list.extend([(join(class_dir, image_dir), idx) for image_dir in img_dirs])
        elif dataset_type in ['train', 'valid', 'test'] or 'test' in dataset_type:
            root_dir = join(root,dataset_type)
            self.class_names = listdir(root_dir)
            self.class_names.sort()
            self.class_num = len(self.class_names)
            
            for idx, item in enumerate(self.class_names):   # idx 相当于类别标号，此处0-nonstandard,1-standard
                class_dir = join(root_dir, item)
                img_dirs = listdir(class_dir)
                # img_dirs.sort()
                self.item_list.extend([(join(class_dir, image_dir), idx) for image_dir in img_dirs])
        else:
            raise ValueError('No dataset type {} , dataset type must in [\'train\',\'test\']'.format(dataset_type))
        
        print('0-{} 1-{}'.format(self.class_names[0],self.class_names[1]))
        
    def get_XY(self):
        x, y = zip(*(self.item_list))
        return x, y

    def __len__(self):
        """返回 dataset 大小"""
        return len(self.item_list)

    def set_transforms(self, tf):
        self.transform = tf

    def __getitem__(self, idx):
        """根据 idx 从 图片名字-类别 列表中获取相应的 image 和 label"""
        img_dir, label = self.item_list[idx]
        name_list = listdir(img_dir)
        tensor_list = []
        for item in name_list:
            img_position, extension = splitext(item)
            if int(img_position) in self.select_list:    # 只读入相应位置的图片
                img = Image.open(join(img_dir,item))
                tensor_list.append(self.transform(img))
        result = cat(tensor_list, dim=0)
        return result, label, img_dir