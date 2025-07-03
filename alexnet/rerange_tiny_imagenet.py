import os
import glob

"""
Move image files to agree with directory of form : /root/[class]/[img_id].jpeg
This is required to use torchvision.ImageFolder for convenience.

Currently, tiny imagenet has directory of form : /root/[class]/images/[img_id].jpeg
We need to move image files to its parent directory.
"""


'''
这段代码的目的是重新组织图像文件的目录结构，
以便于使用 PyTorch 的 torchvision.ImageFolder 类来方便地加载数据。
这个类要求每个类别的图像存放在对应的子目录中。
这里的代码片段是为了从 Tiny ImageNet 数据集的一个现有结构调整到另一个结构。
具体的操作包括移动图像文件并删除某些文件。下面是对这段代码的详细逐行解释
'''

IMAGENET_DIR = 'tiny-imagenet-200'

for root, dirs, files in os.walk(IMAGENET_DIR): # os.walk() 生成当前目录下的文件树中的文件和目录名
    if 'train' in root and 'images' in root:
        class_dir, _ = os.path.split(root) # os.path.split(root) 用于获取当前目录的上级目录路径，即类别的目录
        print('moving for : {}'.format(class_dir))

        # remove annotation files
        for txtfile in glob.glob(os.path.join(class_dir, '*.txt')):
            os.remove(txtfile)

        # move image files to parent directory
        for img_file in files:
            original_path = os.path.join(root, img_file)
            new_path = os.path.join(class_dir, img_file)
            os.rename(original_path, new_path)
        os.rmdir(root)