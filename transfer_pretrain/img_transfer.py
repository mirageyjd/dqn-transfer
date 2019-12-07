from __future__ import print_function
from torch.autograd import Variable
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


pong_mean_img = Image.open('./transfer_pretrain/pong_mean.png')
pong_mean_img.load()
pong_mean_img = np.asarray(pong_mean_img, dtype="float")


def pong_img_preprocess(img):
    img = img.astype(float)
    # Rotate
    img -= pong_mean_img
    img[img > 255] = 255
    img[img < 0] = 0
    # Crop the score
    for i in range(10):
        img[i,:] = np.zeros((84))
        img[(83-i),:] = np.zeros((84))
    img = np.rot90(img)
    img = img.astype('uint8')
    return img


def img_transfer(encode, decode, transform, img):
    '''
        img shape: (4, 84, 84)
    '''
    ret = np.zeros((4, 84, 84))
    for i in range(img.shape[0]):
        single_img = img[i]
        single_img = pong_img_preprocess(single_img)
        reformat_img = np.zeros((84, 84, 3)).astype('uint8')
        for j in range(3):
            reformat_img[:,:,j] = single_img
        reformat_img = Image.fromarray(reformat_img)
        outputs = img_transfer_helper(encode, decode, transform, reformat_img)
        raw_data = outputs.cpu().data.numpy()
        img_cpu = raw_data / np.max(raw_data) * 255.
        ret[i,:,:] = img_cpu[0][0].copy()
    return ret


def img_transfer_helper(encode, decode, transform, img):
    with torch.no_grad():
        image = Variable(transform(img).unsqueeze(0).cuda())
        # Start testing
        content, _ = encode(image)
        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        return outputs


def load_model(config_path, model_path, unit_path):
    '''
        config_path: file system path to the config file
        model_path:  file system path to pretrained model (generator)
    '''
    # Change the path to UNIT folder
    sys.path.insert(0, unit_path)
    from utils import get_config, pytorch03_to_pytorch04
    from trainer import UNIT_Trainer
    config = get_config(config_path)
    trainer = UNIT_Trainer(config)
    try:
        state_dict = torch.load(model_path)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(model_path))
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.cuda()
    trainer.eval()
    # Transfer from Pong to Tennis, namely b2a
    encode = trainer.gen_b.encode # encode function
    decode = trainer.gen_a.decode # decode function
    sys.path.remove(unit_path)
    # Transform
    new_size = config['new_size']
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return encode, decode, transform


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    # Load model
    encode, decode, transform = load_model('../configs/unit_atari_folder.yaml',
                                           './unit_gan_model/gen_00450000.pt',
                                           './UNIT')
    # img = Image.open('../datasets/atari/testB/pong6.png').convert('RGB')
    img = Image.open('./debug.png')
    img = img_transfer_helper(encode, decode, transform, img)
    raw_data = img.cpu().data.numpy()
    img_cpu = raw_data / np.max(raw_data) * 255.
    img_cpu = img.cpu().data.numpy() * 255.
    img = Image.fromarray(img_cpu[0][0])
    img.show()
