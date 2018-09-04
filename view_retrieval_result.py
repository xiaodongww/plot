import os
import torch
torch.cuda.set_device(3)

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from mAP import mAP
from imagenet import imagenet
import time

def load_state_dict(model, path):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model

net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512, 96)
net.cuda()
# net = load_state_dict(net, "/home/wuxiaodong/hashing2/DSH29/snapshots/train15_2.pytorch")
net = load_state_dict(net, "/home/wuxiaodong/hashing2/DSH24/snapshots/train3_13.pytorch")

# ------------------------prepare data loader Start---------------------------------------#
train_path = '/home/hechen/ILSVRC/ILSVRC2012_img_train/'
val_path = '/home/hechen/ILSVRC/ILSVRC2012_img_val/'

transformations = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data = imagenet(val_path, train='val_imgs.txt', transform=transformations, resize=None, tree=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=200, shuffle=False, sampler=None,
                                           num_workers=8, pin_memory=False)
val_ori_data = imagenet(val_path, train='val_imgs.txt', transform=transforms.ToTensor(), resize=None, tree=True)
# ------------------------prepare data loader END---------------------------------------#

def gen_dataset(net, loader):
    net.eval()
    outputs = []
    labels = []
    for batch_idx, (data, target, ancestor1, ancestor2, ancestor3)in enumerate(loader):
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        ancestor1 = Variable(ancestor1.cuda())
        ancestor2 = Variable(ancestor2.cuda())
        ancestor3 = Variable(ancestor3.cuda())


        output = net(data)

        outputs += output.data.cpu().tolist()
        labels += ancestor1.data.cpu().tolist()

    bicodes = torch.sign(torch.Tensor(outputs)).cpu().numpy()
    labels = torch.Tensor(labels).cpu().numpy()
    return bicodes, labels

bicodes, labels = gen_dataset(net, valloader)

class labelConverter():
    def __init__(self, meta_path="/home/hechen/ILSVRC/ILSVRC2012_devkit_t12/data/meta.mat"):
        import scipy.io as sio
        synsets  = sio.loadmat(meta_path)['synsets']
        height_count = {i:[] for i in range(20)}
        new_synsets = {}
        wnid_to_ilsid = {}
        for synset in synsets:
            ils_id = synset[0][0][0][0]
            wordnet_height = synset[0][-2][0][0]
            children = synset[0][-3][0]
            num_children = len(children)
            words = synset[0][2][0]
            wnid = synset[0][1][0]
            height_count[wordnet_height].append(ils_id)
            new_synsets[ils_id] = {}
            new_synsets[ils_id]['wnid'] = wnid
            new_synsets[ils_id]['words'] = words
            new_synsets[ils_id]['wordnet_height'] = wordnet_height
            new_synsets[ils_id]['num_children'] = num_children
            new_synsets[ils_id]['children'] = children
            wnid_to_ilsid[wnid] = ils_id
        self.new_synsets = new_synsets
    def label_to_word(self, label):
        return self.new_synsets[label]['words']
converter = labelConverter()
# print(converter.label_to_word(13))

def query(q, db, k):
    """
    return nearest idxs
    """
    codelen = db.shape[1]
    dist = 0.5 * (codelen - np.dot(q, db.transpose()))
    return dist.argsort()[:k]

def gen_code(net, dataset, idx):
    data = Variable(torch.stack([dataset[idx][0]]).cuda(), volatile=True)
    output = net(data).squeeze()
    output = torch.sign(output).data.cpu().numpy()
    return output

def plot_img(img):
    toimg = transforms.ToPILImage('RGB')
    img = toimg(img.squeeze())
    img.show()

def add_text(img, truth, label):
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont
    font_path = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
    font = ImageFont.truetype(font_path, 18)
    blank = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    blank.paste(img, (0, 0))
    d = ImageDraw.Draw(blank)

    d.text([20, 5], converter.label_to_word(label[0]), 'blue' if truth[0]==label[0] else 'red', font=font) 
    d.text([20, 23], converter.label_to_word(label[1]), 'blue' if truth[1]==label[1] else 'red', font=font) 
    d.text([20, 40], converter.label_to_word(label[2]), 'blue' if truth[2]==label[2] else 'red', font=font) 
    d.text([20, 58], converter.label_to_word(label[3]), 'blue' if truth[3]==label[3] else 'red', font=font) 
    
    return blank
# toimg = transforms.ToPILImage('RGB')
# img = toimg(val_ori_data[901][0].squeeze()).resize((128, 128)).convert('RGBA')
# new_img = add_text(img, '123123123')
# new_img.show()

def plot_query(idx):
    from PIL import Image
    q = gen_code(net, val_data, idx)

    topk = query(q, bicodes, 100)
    toimg = transforms.ToPILImage('RGB')

    background = Image.new('RGBA', (128*11, 1280), (0, 0, 0, 0))
    q_img = toimg(val_ori_data[idx][0].squeeze()).resize((128, 128)).convert('RGBA')
    q_img = add_text(q_img, val_ori_data[idx][1:], val_ori_data[idx][1:])
    background.paste(q_img, (0,0,128,128))
    for i in range(100):
        img = toimg(val_ori_data[topk[i]][0].squeeze()).resize((128, 128)).convert('RGBA')
        img = add_text(img, val_ori_data[idx][1:], val_ori_data[topk[i]][1:])
        background.paste(img, (i%10*128+128, i//10*128, i%10*128+128+128, i//10*128+128))
        # print(128*divmod(i, 10)[1], i%10*128)
         # 128*divmod(i, 10)[1]+128, i%10*128+128))
    
    background.show()

def multi_grain_query(idx, start, end, level):
    from PIL import Image
    # q = gen_code(net, val_data, idx)

    topk = query(bicodes[idx, start:end], bicodes[:, start:end], 100)
    multi_labels = []
    paths = []
    for i in range(100):
        labels = [converter.label_to_word(label) for label in val_ori_data[topk[i]][1:]] 
        multi_labels.append(labels[4-level])
        paths.append(os.path.join(val_ori_data.root, val_ori_data.part_paths[topk[i]]))
    return multi_labels, paths

query_id = 786
query_label = 
level1_mlabels, level1_path = multi_grain_query(query_id, 0, 12, 1)
level2_mlabels, level2_path= multi_grain_query(query_id, 0, 24, 2)
level3_mlabels, level3_path= multi_grain_query(query_id, 0, 48, 3)
level4_mlabels, level4_path= multi_grain_query(query_id, 0, 96, 4)


# import random

# plot_query(random.choice(range(len(val_data))))
plot_query(query_id)
