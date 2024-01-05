from cgi import test
from requests import get
import torch
import unittest
from src.adv_attack.attack.Attack import adversarial_attack
import os
from src.siamese import *
import torch.utils.data as data
from PIL import Image, ImageOps
import pickle
import numpy as np
from torchvision import transforms

class GetLoader(data.Dataset):
    def __init__(self, data_root, label_dict, transform=None, grayscale=False):
        
        self.transform = transform
        self.data_root = data_root
        self.grayscale = grayscale

        with open(label_dict, 'rb') as handle:
            self.label_dict = pickle.load(handle)

        self.classes = list(self.label_dict.keys())

        self.n_data = len(os.listdir(self.data_root))

        self.img_paths = []
        self.labels = []

        for data in os.listdir(self.data_root):
            image_path = data
            label = image_path.split('+')[0]
            
            # deal with inconsistency in naming 
            if brand_converter(label) == 'Microsoft':
                self.labels.append(label)
                
            elif brand_converter(label) == 'DHL Airways':
                self.labels.append('DHL')
                
            elif brand_converter(label) == 'DGI French Tax Authority':
                self.labels.append('DGI (French Tax Authority)')
                
            else:
                self.labels.append(brand_converter(label))

            self.img_paths.append(image_path)

    def __getitem__(self, item):

        img_path, label= self.img_paths[item], self.labels[item]
        img_path_full = os.path.join(self.data_root, img_path)
        
        if self.grayscale:
            img = Image.open(img_path_full).convert('L').convert('RGB')
        else:
            img = Image.open(img_path_full).convert('RGB')

        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

        # label = np.array(label,dtype='float32')
        label = self.label_dict[label]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.n_data

class TestAttack(unittest.TestCase):
    def setUp(self):
        self.model = self.get_model()
        self.test_set = GetLoader(data_root='./datasets/test/gaussian_noise/', 
                    label_dict='./datasets/test/gaussian_noise.json')
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model(self):
        model = KNOWN_MODELS["BiT-M-R50x1"](head_size=277, zero_head=True)
        checkpoint = torch.load('./src/phishpedia/resnetv2_rgb_new.pth.tar', 
                                map_location="cpu")["model"]

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        model.to(self.device)
        model.eval()
        return model
    
    def compute_acc(dataloader, model, device):
        correct = 0
        total = 0

        for b, (x, y) in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                pred_cls = torch.argmax(logits, dim=1)

                correct += torch.sum(torch.eq(pred_cls, y)).item()
                total += y.shape[0]
                
        print('Accuracy after changing relu function: {:.2f}'.format(correct/total))    
        return correct/total
    
    def compute_acc_after_attack(self, attack_method):
        check = adversarial_attack(method=attack_method, model=self.model, dataloader=self.test_loader, 
                           device=self.device, num_classes=277, save_data=True)
        acc, _ = check.batch_attack()
        return acc

    def compare_acc(self, attack_method):
        origin_acc = self.compute_acc(self.test_loader, self.model, self.device)
        acc = self.compute_acc_after_attack(attack_method)
        self.assertLess(acc, origin_acc)
    
    def test_batch_attack(self):
        self.compute_acc_after_attack('fgsm')
        self.compute_acc_after_attack('stepll')
        self.compute_acc_after_attack('jsma')
        self.compute_acc_after_attack('deepfool')
        self.compute_acc_after_attack('cw')

if __name__ == '__main__':
    unittest.main()