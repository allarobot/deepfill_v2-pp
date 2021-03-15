import os
import time
import datetime
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import dataset
import utils

suffix_setting = ['.g','.d','.optg','.optd']
def save_model(nets, epoch, opt):
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = 'deepfillv2_%s_epoch%d' % (opt.gan_type, epoch)
    model_name = os.path.join(opt.save_path, model_name)
    if epoch % opt.checkpoint_interval == 0:
        for i,net in enumerate(nets):
            suffix = suffix_setting[i]
            paddle.save(net.state_dict(), model_name+suffix)
    print('The trained model is successfully saved at epoch %d' % (epoch))

def load_model(opt):
    """Save the model at "checkpoint_interval" and its multiple"""
    nets = []
    for suffix in suffix_setting:
        model_name = opt.load_name + suffix
        full_model_name = os.path.join(opt.save_path, model_name)
        if os.path.exists(full_model_name):
            net = paddle.load(full_model_name)
            nets.append(net)
    if nets is []:
        print('no pretrain model found')
    else:
        print('The trained model is successfully loaded')
    return nets

def resume_model(nets,opt):
    """Save the model at "checkpoint_interval" and its multiple"""
    model_name = 'deepfillv2_%s_epoch%d' % (opt.gan_type, opt.resume_epoch)
    fully_resumed = True
    for i,suffix in enumerate(suffix_setting):
        full_model_name = opt.load_name + suffix
        abs_model_name = os.path.join(opt.save_path, full_model_name)
        if os.path.exists(abs_model_name):
            state_dict = paddle.load(abs_model_name)
            nets[i].set_state_dict(state_dict)
            print(f'The trained model:{model_name} is reloaded successfully')
        else:
            print(f'The trained model:{model_name} is not found')
            fully_resumed = False
    return fully_resumed

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    scheduler_g = paddle.optimizer.lr.ExponentialDecay(learning_rate=opt.lr_g, gamma=opt.lr_decrease_factor,
                                                       verbose=True)
    scheduler_d = paddle.optimizer.lr.ExponentialDecay(learning_rate=opt.lr_d, gamma=opt.lr_decrease_factor,
                                                       verbose=True)

    optimizer_g = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=scheduler_g, beta1=opt.b1,
                                        beta2=opt.b2, weight_decay=opt.weight_decay)
    optimizer_d = paddle.optimizer.Adam(parameters=discriminator.parameters(), learning_rate=scheduler_d, beta1=opt.b1,
                                        beta2=opt.b2, weight_decay=opt.weight_decay)
    start_epoch = 0
    if opt.resume_epoch>0:
        nets = [generator,optimizer_g,discriminator,optimizer_d]
        ret = resume_model(nets,opt)
        if ret:
            start_epoch = opt.resume_epoch

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    generator.train()
    discriminator.train()
    for epoch in range(start_epoch,opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):
            ### Train Discriminator

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img = paddle.to_tensor(img)
            mask = paddle.to_tensor(mask)

            generator.eval()
            discriminator.train()
            optimizer_d.clear_grad()
            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(img, mask)

            # Overall Loss and optimize
            loss_D = - paddle.mean(true_scalar) + paddle.mean(fake_scalar)

            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            generator.train()
            discriminator.eval()
            optimizer_g.clear_grad()

            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)

            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = - paddle.mean(fake_scalar)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)  # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                   opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                  ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.numpy(),
                   second_MaskL1Loss.numpy()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                  (loss_D.numpy(), GAN_Loss.numpy(), second_PerceptualLoss.numpy(), time_left))

        # Learning rate decrease
        if (epoch+1) % opt.lr_decrease_epoch == 0:
            scheduler_g.step()
            scheduler_d.step()

        # Save the model
        nets = [generator,optimizer_g,discriminator,optimizer_d]
        save_model(nets, (epoch + 1), opt)

        ### Sample data every epoch
        masked_img = img * (1 - mask) + mask
        mask = paddle.concat((mask, mask, mask), 1)
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out,second_out_wholeimg]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out','second_out_wholeimg']
            utils.save_sample_png(sample_folder=sample_folder, sample_name='epoch%d' % (epoch + 1), img_list=img_list,
                                  name_list=name_list, pixel_max_cnt=255)

def LSGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # Optimizers
    scheduler_g = paddle.optimizer.lr.ExponentialDecay(learning_rate=opt.lr_g, gamma=opt.lr_decrease_factor)
    scheduler_d = paddle.optimizer.lr.ExponentialDecay(learning_rate=opt.lr_d, gamma=opt.lr_decrease_factor)
    optimizer_g = paddle.optimizer.Adam(parameters=generator.parameters(), learning_rate=scheduler_g, beta1=opt.b1,
                                        beta2=opt.b2, weight_decay=opt.weight_decay)
    optimizer_d = paddle.optimizer.Adam(parameters=discriminator.parameters(), learning_rate=scheduler_d, beta1=opt.b1,
                                        beta2=opt.b2, weight_decay=opt.weight_decay)

    start_epoch = 0
    if opt.resume_epoch>0:
        nets = [generator,optimizer_g,discriminator,optimizer_d]
        ret = resume_model(nets,opt)
        if ret:
            start_epoch = opt.resume_epoch

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(start_epoch, opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):

            # LSGAN vectors
            img = paddle.to_tensor(img)
            mask = paddle.to_tensor(mask)
            valid = paddle.to_tensor(np.ones((img.shape[0], 1, 8, 8)),dtype='float32')
            fake = paddle.to_tensor(np.zeros((img.shape[0], 1, 8, 8)),dtype='float32')


            ### Train Discriminator
            generator.eval()
            discriminator.train()
            optimizer_d.clear_grad()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(img, mask)

            # Overall Loss and optimize
            loss_fake = MSELoss(fake_scalar, fake)
            loss_true = MSELoss(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            generator.train()
            discriminator.eval()
            optimizer_g.clear_grad()

            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)

            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = MSELoss(fake_scalar, valid)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)  # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + \
                   opt.lambda_perceptual * second_PerceptualLoss + opt.lambda_gan * GAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                  ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.numpy(),
                   second_MaskL1Loss.numpy()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" %
                  (loss_D.numpy(), GAN_Loss.numpy(), second_PerceptualLoss.numpy(), time_left))

        # Learning rate decrease
        if (epoch+1) % opt.lr_decrease_epoch == 0:
            scheduler_g.step()
            scheduler_d.step()
        # adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        # adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        nets = [generator,optimizer_g,discriminator,optimizer_d]
        save_model(nets, (epoch + 1), opt)
        ### Sample data every epoch
        masked_img = img * (1 - mask) + mask
        mask = paddle.concat((mask, mask, mask), 1)
        if (epoch + 1) % 1 == 0:
            img_list = [img, mask, masked_img, first_out, second_out,second_out_wholeimg]
            name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out','second_out_wholeimg']
            utils.save_sample_png(sample_folder=sample_folder, sample_name='epoch%d' % (epoch + 1), img_list=img_list,
                                  name_list=name_list, pixel_max_cnt=255)
