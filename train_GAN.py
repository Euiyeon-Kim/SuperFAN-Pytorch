import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.vgg import vgg19
import cv2
import os
import numpy as np
import sys

from utils import Dataset
from utils import evaluation_dataset
from utils import save_sample
from utils import renormalization

# from model import Generator
# from model import Discriminator


from model import *
from utils import eval, compute_gradient_penalty
import tensorboardX


# proj_directory = '/project'
# data_directory = '/dataset'
#
# menpo = "/dataset/LS3D-W/Menpo-3D"
# celeba = '/dataset/img_align_celeba'
# _300w = "/dataset/LS3D-W/300W-Testset-3D"
# aflw = "/dataset/LS3D-W/AFLW2000-3D-Reannotated"
# ffhq = '/dataset/ffhq'
# afad = '/dataset/AFAD-SR'

# settings for local
proj_directory = './'
data_directory = '../'

ffhq = '../ffhq'
afad = '../AFAD-SR'

save_path = os.path.join(proj_directory, 'models')
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_G = os.path.join(save_path, 'generator.pth')
# save_path_D = os.path.join(save_path, 'discriminator.pth')

batch_size = 4
vgg = vgg19(pretrained=True).eval()

def load_FAN():
    '''
    input: (256, 256, 3) image with scale [0, 1]
    output: (256, 256, 68) face landmarks
    '''
    from FAN.models import FAN
    import torch.utils.model_zoo as model_zoo
    print('===> Loading FAN model')
    model = FAN(2)

    weights = model_zoo.load_url('https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar', map_location=lambda storage, loc: storage)

    #cutoff two hourglass network
    pretrained_dict = {k: v for k, v in weights.items() if k in model.state_dict()}

    model.load_state_dict(pretrained_dict)
    model.eval()

    return model


def train(train_directories, epoch):
    print('start')

    # add summary
    if not os.path.exists(os.path.join(proj_directory, 'logs')):
        os.makedirs(os.path.join(proj_directory, 'logs'))
    summary_writer = tensorboardX.SummaryWriter('./logs/')

    dataset = Dataset(train_directories, also_valid=True)
    loaded_training_data = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, num_workers=12
    )

    valid_dataset = dataset.clone_for_validation()  # evaluation dataset
    loaded_valid_data = DataLoader(dataset=valid_dataset, batch_size=1)

    vgg_feature = nn.Sequential(*list(vgg.features)[:-1]).cuda()
    FAN = load_FAN().cuda()
    preprocess_for_FAN = upsample().cuda()
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()


    if os.path.exists(save_path_G):
        generator.load_state_dict(torch.load(save_path_G))
        print('reading generator checkpoints...')
    # if os.path.exists(save_path_D):
    #     discriminator.load_state_dict(torch.load(save_path_D))
    #     print('reading discriminator checkpoints...')
    if not os.path.exists(os.path.join(proj_directory, 'validation')):
        os.makedirs(os.path.join(proj_directory, 'validation'))

    mse = nn.MSELoss().cuda()

    D_start = 60
    learning_rate = 2.5e-4
    final_lr = 1e-5
    decay = (learning_rate - final_lr) / D_start

    print('train with MSE and perceptual loss')
    for epoch in range(epoch):
        # G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        G_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)
        D_optimizer = optim.RMSprop(discriminator.parameters(), lr=learning_rate)

        for i, data in enumerate(loaded_training_data):
            lr, gt, _ = data
            gt = gt.float()
            lr = lr.cuda()
            gt = gt.cuda()

            ##### G step #####
            # forwarding
            sr = generator(lr)

            # initialization
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
            #################


            ##### D step #####
            if epoch >= D_start:
                D_optimizer.zero_grad()

                sr = generator(lr).detach()
                fake_logit = discriminator(sr).mean()
                real_logit = discriminator(gt).mean()

                gradient_penalty = compute_gradient_penalty(discriminator, gt, sr)
                d_loss = fake_logit - real_logit + 10. * gradient_penalty

                d_loss.backward()
                D_optimizer.step()
            #################

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
            validation = os.path.join(proj_directory, 'validation', str(epoch))
            os.makedirs(validation)

            total_mse = 0
            total_ssim =0
            total_psnr = 0

            for _, val_data in enumerate(loaded_valid_data):
                lr, gt, img_name = val_data
                sr = generator(lr)

                # for evaluating images
                mse, ssim, psnr = eval(gt.data.cpu().numpy(), sr.data.cpu().numpy())
                total_mse += mse / len(loaded_valid_data)
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
            learning_rate = 2.5e-4
        else:
            learning_rate -= decay

    ######################################################
    #                    train GAN                       #
    ######################################################

    # save checkpoints
    torch.save(generator.state_dict(), save_path_G)
    # torch.save(discriminator.state_dict(), save_path_D)
    print('training finished.')


if __name__ == '__main__':
    epochs = 70
    train_directories = [afad]
    train(train_directories, epochs)
