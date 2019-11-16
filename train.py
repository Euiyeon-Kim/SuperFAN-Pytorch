import os
import cv2
import sys
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19
import tensorboardX

from model import *
from utils import Dataset
from utils import save_sample
from utils import renormalization
from utils import evaluation_dataset
from utils import eval, compute_gradient_penalty

conf = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
train_conf = conf['train']
test_conf = conf['test']

project_dir = conf['project_dir']
dataset_root = conf['dataset_root']
dataset = os.path.join(dataset_root, conf['dataset_name'])

save_dir = conf['save_dir']
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path_G = os.path.join(save_dir, 'generator.pth')
save_path_D = os.path.join(save_dir, 'discriminator.pth')

batch_size = train_conf['batch_size']
vgg = vgg19(pretrained=True).eval()

def load_FAN(num_modules):
    '''
        input: (256, 256, 3) image with scale [0, 1]
        output: (256, 256, 68) face landmarks
    '''
    from FAN.model import FAN
    import torch.utils.model_zoo as model_zoo
    print('===> Loading FAN model')
    model = FAN(num_modules)

    weights = model_zoo.load_url('https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar', map_location=lambda storage, loc: storage)

    #cutoff two hourglass network
    pretrained_dict = {k: v for k, v in weights.items() if k in model.state_dict()}

    model.load_state_dict(pretrained_dict)
    model.eval()

    return model

def train(train_directories, epoch):
    print('start')

    # Logging information
    if not os.path.exists(os.path.join(project_dir, 'logs')):
        os.makedirs(os.path.join(project_dir, 'logs'))
    summary_writer = tensorboardX.SummaryWriter('./logs/')

    # Dataloader
    dataset = Dataset(train_directories, also_valid=True)
    loaded_training_data = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, num_workers=12
    )
    valid_dataset = dataset.clone_for_validation()
    loaded_valid_data = DataLoader(dataset=valid_dataset, batch_size=1)

    # Loading SuperFAN model
    vgg_feature = nn.Sequential(*list(vgg.features)[:-1]).cuda()
    FAN = load_FAN(train_conf['num_FAN_modules']).cuda()
    preprocess_for_FAN = upsample().cuda()
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    mse = nn.MSELoss().cuda()

    if os.path.exists(save_path_G):
        generator.load_state_dict(torch.load(save_path_G))
        print('reading generator checkpoints...')
    if os.path.exists(save_path_D):
        discriminator.load_state_dict(torch.load(save_path_D))
        print('reading discriminator checkpoints...')
    if not os.path.exists(os.path.join(project_dir, 'validation')):
        os.makedirs(os.path.join(project_dir, 'validation'))

    # Setting learning rate decay
    D_start = train_conf['start_decay']
    learning_rate = train_conf['start_lr']
    final_lr = train_conf['final_lr']
    decay = (learning_rate - final_lr) / D_start

    print('train with MSE and perceptual loss')
    for epoch in range(epoch):

        G_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
        D_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

        for i, data in enumerate(loaded_training_data):
            lr, gt, _ = data
            gt = gt.float()
            lr = lr.cuda()
            gt = gt.cuda()

            #========= Training G =========#
            # Forwarding
            sr = generator(lr)

            # Initialization
            G_optimizer.zero_grad()

            # vgg loss
            mse_loss = mse(sr, gt)
            sr_vgg = vgg_feature(sr)
            gt_vgg = vgg_feature(gt)
            vgg_loss = mse(sr_vgg, gt_vgg) * 0.006

            sr_FAN = FAN(preprocess_for_FAN(sr))[-1]
            gt_FAN = FAN(preprocess_for_FAN(gt))[-1].detach()

            FAN_loss = mse(sr_FAN, gt_FAN)

            g_loss = mse_loss + vgg_loss + FAN_loss
            if epoch >= D_start:
                fake_logit = discriminator(sr).mean()
                G_adv_loss = - fake_logit
                g_loss += G_adv_loss * 1e-3

            g_loss.backward()
            G_optimizer.step()


            #========= Training D =========#
            if epoch >= D_start:
                D_optimizer.zero_grad()

                sr = generator(lr).detach()
                fake_logit = discriminator(sr).mean()
                real_logit = discriminator(gt).mean()

                gradient_penalty = compute_gradient_penalty(discriminator, gt, sr)
                d_loss = fake_logit - real_logit + 10. * gradient_penalty

                d_loss.backward()
                D_optimizer.step()


            if i % 10 == 0:
                print("loss at %d : %d ==>\tmse: %.6f  vgg: %.6f  FAN: %.6f" % (epoch, i, mse_loss, vgg_loss, FAN_loss))
                summary_writer.add_scalar('mse_loss', mse_loss.data.cpu().numpy(), epoch * len(loaded_training_data) + i)
                summary_writer.add_scalar('vgg_loss', vgg_loss.data.cpu().numpy(), epoch * len(loaded_training_data) + i)
                summary_writer.add_scalar('FAN_loss', FAN_loss.data.cpu().numpy(), epoch * len(loaded_training_data) + i)

                if epoch >= D_start:
                    print("\t\t\t\tD_loss: %.6f  G_loss: %.6f" % (d_loss.data.cpu(), G_adv_loss.data.cpu()))
                    summary_writer.add_scalar('D_loss', d_loss.data.cpu().numpy(), epoch * len(loaded_training_data) + i)
                    summary_writer.add_scalar('G_loss', G_adv_loss.data.cpu().numpy(),
                                              epoch * len(loaded_training_data) + i)

        if epoch % 1 == 0:
            validation = os.path.join(project_dir, 'validation', str(epoch))
            os.makedirs(validation)

            total_mse = 0
            total_ssim =0
            total_psnr = 0

            for _, val_data in enumerate(loaded_valid_data):
                lr, gt, img_name = val_data
                sr = generator(lr)

                # for evaluating images
                MSE, ssim, psnr = eval(gt.data.cpu().numpy(), sr.data.cpu().numpy())
                total_mse += MSE / len(loaded_valid_data)
                total_ssim += ssim / len(loaded_valid_data)
                total_psnr += psnr / len(loaded_valid_data)

                # for saving images
                sr = sr[0]
                sr = renormalization(sr)
                sr = sr.cpu().detach().numpy()
                sr = sr.transpose(1, 2, 0)
                img_name = img_name[0]

                filename = os.path.join(validation, img_name + '.png')
                cv2.imwrite(filename=filename, img=sr)

            # save logs
            summary_writer.add_scalar('valid/mse', total_mse, epoch)
            summary_writer.add_scalar('valid/ssim', total_ssim, epoch)
            summary_writer.add_scalar('valid/psnr', total_psnr, epoch)

            # save checkpoints
            torch.save(generator.state_dict(), save_path_G)
            # torch.save(discriminator.state_dict(), save_path_D)

        # decay learning rate after one epoch
        if epoch >= D_start:
            learning_rate = train_conf['start_lr']
        else:
            learning_rate -= decay

    # Save models
    torch.save(generator.state_dict(), save_path_G)
    torch.save(discriminator.state_dict(), save_path_D)
    print('training finished.')


if __name__ == '__main__':
    epochs = train_conf['epochs']
    train_directories = [dataset]
    train(train_directories, epochs)
