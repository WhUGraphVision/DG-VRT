from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET,G2_DCGAN,G2_DCGAN_1,G3_DCGAN
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss,generator2_loss,generator1_loss,generator3_loss
import os
import time
import numpy as np
import sys

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        num=3
        for i in range(num):
            temp=[]
            netsD.append(temp)
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G3_DCGAN() ## imoort GAN Model
            for i in range(num):
                #tempnetsD=[D_NET(b_jcu=False)]
                netsD[i].append(D_NET(b_jcu=False)) 
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                for i in range(num):
                    netsD[i].append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                for i in range(num):
                    netsD[i].append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                for i in range(num):
                    netsD[i].append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(num):
            for j in range(len(netsD[i])):
                netsD[i][j].apply(weights_init)
            # print(netsD[i])
            print('# of netsD', len(netsD[i]))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(num):
                    for j in range(len(netsD[i])):
                        s_tmp = Gname[:Gname.rfind('/')]
                        Dname = '%s/netD%d.pth' % (s_tmp, i)
                        print('Load D from: ', Dname)
                        state_dict = torch.load(Dname, map_location=lambda storage, loc: storage)
                        netsD[i][j].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(num):
                for j in range(len(netsD[i])):
                    netsD[i][j].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num=len(netsD)
        num_Ds = len(netsD[0])
        for i in range(num):
            temp=[]
            optimizersD.append(temp)
        for i in range(num):
            for j in range(num_Ds):
                opt = optim.Adam(netsD[i][j].parameters(),lr=cfg.TRAIN.DISCRIMINATOR_LR,betas=(0.5, 0.999))
                optimizersD[i].append(opt)
  

        optimizerG = optim.Adam(netG.parameters(),lr=cfg.TRAIN.GENERATOR_LR,betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        num_Ds = len(netsD[0])
        for i in range(len(netsD)):
            for j in range(num_Ds):
                netD = netsD[i][j]
                torch.save(netD.state_dict(),'%s/netD_%d_%d.pth' % (self.model_dir, i,j))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise0,noise1, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs1,fake_imgs2, attention_maps1,attention_maps2, _, _ = netG(noise0, noise1,sent_emb, words_embs, mask)
        for i in range(len(attention_maps1)):
            if len(fake_imgs1) > 1:
                img1 = fake_imgs1[i + 1].detach().cpu()
                lr_img1 = fake_imgs1[i].detach().cpu()
                img2 = fake_imgs2[i + 1].detach().cpu()
                lr_img2 = fake_imgs2[i].detach().cpu()
            else:
                img1 = fake_imgs1[0].detach().cpu()
                lr_img1 = None
                img2 = fake_imgs2[0].detach().cpu()
                lr_img2 = None
            attn_maps1 = attention_maps1[i]
            attn_maps2 = attention_maps2[i]
            att_sze1 = attn_maps1.size(2)
            att_sze2 = attn_maps2.size(2)
            img_set1, _ = build_super_images(img1, captions, self.ixtoword,attn_maps1, att_sze1, lr_imgs=lr_img1)
            img_set2, _ = build_super_images(img2, captions, self.ixtoword,attn_maps2, att_sze2, lr_imgs=lr_img2)
            if img_set1 is not None:
                im = Image.fromarray(img_set1)
                fullpath = '%s/G_%s_%d_%d_1.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)
            if img_set2 is not None:
                im = Image.fromarray(img_set2)
                fullpath = '%s/G_%s_%d_%d_2.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        num=2
        for j in range(num):
            if j == 0:
                 fake_imgs=fake_imgs1
                 #att_maps=att_maps1
                 #att_sze=att_sze1
            elif j == 1:
                 fake_imgs=fake_imgs2
                 #att_maps=att_maps2
                 #att_sze=att_sze2
            i = -1
            img = fake_imgs[i].detach()
            region_features, _ = image_encoder(img)
            att_sze = region_features.size(2)
            _, _, att_maps = words_loss(region_features.detach(),words_embs.detach(),None, cap_lens,None, self.batch_size)
            img_set, _ = build_super_images(fake_imgs[i].detach().cpu(),captions, self.ixtoword, att_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/D_%s_%d_%d.png'\
                % (self.image_dir, name, gen_iterations,j+1)
                im.save(fullpath)
    def save_3img_results(self, netG, noise0,noise1, noise2,sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, name='current'):
        # Save images
        fake_imgs0,fake_imgs1,fake_imgs2, attention_maps0,attention_maps1,attention_maps2, _, _, _, _, _, _ = netG(noise0, noise1,noise2,sent_emb, words_embs, mask)
        for i in range(len(attention_maps1)):
            if len(fake_imgs1) > 1:
                img0 = fake_imgs0[i + 1].detach().cpu()
                lr_img0 = fake_imgs0[i].detach().cpu()
                img1 = fake_imgs1[i + 1].detach().cpu()
                lr_img1 = fake_imgs1[i].detach().cpu()
                img2 = fake_imgs2[i + 1].detach().cpu()
                lr_img2 = fake_imgs2[i].detach().cpu()
            else:
                img0 = fake_imgs0[0].detach().cpu()
                lr_img0 = None
                img1 = fake_imgs1[0].detach().cpu()
                lr_img1 = None
                img2 = fake_imgs2[0].detach().cpu()
                lr_img2 = None
            attn_maps0 = attention_maps0[i]
            attn_maps1 = attention_maps1[i]
            attn_maps2 = attention_maps2[i]
            att_sze0 = attn_maps0.size(2)
            att_sze1 = attn_maps1.size(2)
            att_sze2 = attn_maps2.size(2)
            img_set0, _ = build_super_images(img0, captions, self.ixtoword,attn_maps0, att_sze0, lr_imgs=lr_img0)
            img_set1, _ = build_super_images(img1, captions, self.ixtoword,attn_maps1, att_sze1, lr_imgs=lr_img1)
            img_set2, _ = build_super_images(img2, captions, self.ixtoword,attn_maps2, att_sze2, lr_imgs=lr_img2)
            if img_set0 is not None:
                im = Image.fromarray(img_set0)
                fullpath = '%s/G_%s_%d_%d_0.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)
            if img_set1 is not None:
                im = Image.fromarray(img_set1)
                fullpath = '%s/G_%s_%d_%d_1.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)
            if img_set2 is not None:
                im = Image.fromarray(img_set2)
                fullpath = '%s/G_%s_%d_%d_2.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        num=3
        for j in range(num):
            if j == 0:
                fake_imgs=fake_imgs0
                 #att_maps=att_maps1
                 #att_sze=att_sze1
            elif j == 1:
                 ake_imgs=fake_imgs1
            elif j==2:
                fake_imgs=fake_imgs2
                 #att_maps=att_maps2
                 #att_sze=att_sze2
            i = -1
            img = fake_imgs[i].detach()
            region_features, _ = image_encoder(img)
            att_sze = region_features.size(2)
            _, _, att_maps = words_loss(region_features.detach(),words_embs.detach(),None, cap_lens,None, self.batch_size)
            img_set, _ = build_super_images(fake_imgs[i].detach().cpu(),captions, self.ixtoword, att_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/D_%s_%d_%d.png'\
                % (self.image_dir, name, gen_iterations,j)
                im.save(fullpath)

    def train(self):
        num=3
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise0 = Variable(torch.FloatTensor(batch_size, nz))
        noise1 = Variable(torch.FloatTensor(batch_size, nz))
        noise2 = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise0 = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        fixed_noise1 = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        fixed_noise2 = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise1,noise2, fixed_noise0 = noise1.cuda(), noise2.cuda(),fixed_noise0.cuda()
            noise0,fixed_noise2, fixed_noise1 = noise0.cuda(), fixed_noise2.cuda(),fixed_noise1.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise0.data.normal_(0, 1)
                noise1.data.normal_(0, 1)
                noise2.data.normal_(0, 1)
                fake_imgs0,fake_imgs1,fake_imgs2, _,_, _,mu0, logvar0,mu1, logvar1,mu2, logvar2,= netG(noise0,noise1,noise2, sent_emb, words_embs, mask)
               
                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                t_num=len(netsD[-1])
                for j in range(t_num):
                    for i in range(num):
                        netsD[i][j].zero_grad()
                        errD0 = discriminator_loss(netsD[i][j], imgs[j], fake_imgs0[j],sent_emb, real_labels, fake_labels)
                        errD1 = discriminator_loss(netsD[i][j], imgs[j], fake_imgs1[j],sent_emb, real_labels, fake_labels)
                        errD2 = discriminator_loss(netsD[i][j], imgs[j], fake_imgs2[j],sent_emb, real_labels, fake_labels)
                        errD=errD0+errD1+errD2
                        errD.backward()
                        optimizersD[i][j].step()



                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = generator3_loss(netsD, image_encoder, fake_imgs0,fake_imgs1, fake_imgs2,noise0,noise1,noise2,real_labels,words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss = KL_loss(mu0, logvar0)+KL_loss(mu1, logvar1)+KL_loss(mu2, logvar2)
                #errG_total=errG_total1+errG_total2
                #G_logs=G_logs1+G_logs2
                errG_total += kl_loss
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                # save images
                if gen_iterations % 10000 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_3img_results(netG, fixed_noise0, fixed_noise1,fixed_noise2, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, name='average')
                    load_params(netG, backup_para)
                    #
                    # self.save_img_results(netG, fixed_noise, sent_emb,
                    #                       words_embs, mask, image_encoder,
                    #                       captions, cap_lens,
                    #                       epoch, name='current')
            end_t = time.time()

            #print('''[%d/%d][%d]
            #      Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
            #      % (epoch, self.max_epoch, self.num_batches,
            #         errD_total.item(), errG_total.item(),
            #         end_t - start_t))
            print('''[%d/%d][%d] Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, epoch)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G2_DCGAN_1()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM

            

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            ppp=0
            pppp=1
            #for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for nn in range(5):
                with torch.no_grad():
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    noise = noise.cuda()
                for dl in range(10): 
                    #for step, data in enumerate(self.data_loader, dl):
                    for step, data in enumerate(self.data_loader, dl):
                        cnt += batch_size
                        if step % 100 == 0:
                            print('step: ', step)
                        # if step > 50:
                        #     break
                        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                        #print(captions)
                        hidden = text_encoder.init_hidden(batch_size)
                        # words_embs: batch_size x nef x seq_len
                        # sent_emb: batch_size x nef
                        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                        mask = (captions == 0)
                        num_words = words_embs.size(2)
                        if mask.size(1) > num_words:
                            mask = mask[:, :num_words]

                        #######################################################
                        # (2) Generate fake images
                        ######################################################
                        noise.data.normal_(0, 1)
                        #fake_imgs, _, _, _ = netG(noise, sent_emb, words_embs, mask)
                        fake_imgs1,fake_imgs2, attention_maps1,attention_maps2, _, _ = netG(noise,noise, sent_emb, words_embs, mask)
                        # G attention
                        cap_lens_np = cap_lens.cpu().data.numpy()
                        for j in range(batch_size):
                            s_tmp = '%s/single/%s' % (save_dir, keys[j])
                            folder = s_tmp[:s_tmp.rfind('/')]
                            if not os.path.isdir(folder):
                                print('Make a new folder: ', folder)
                                mkdir_p(folder)
                            #k = -1
                            #print(len(fake_imgs1))
                            for k in range(len(fake_imgs1)):
                                im = fake_imgs1[k][j].data.cpu().numpy()
                            # [-1, 1] --> [0, 255]
                                im = (im + 1.0) * 127.5
                                im = im.astype(np.uint8)
                                im = np.transpose(im, (1, 2, 0))
                                im = Image.fromarray(im)
                                fullpath = '%s_s%d_g0_n%d.png' % (s_tmp, dl,nn)
                                im.save(fullpath)
                        
                        for j in range(batch_size):
                            s_tmp = '%s/single/%s' % (save_dir, keys[j])
                            folder = s_tmp[:s_tmp.rfind('/')]
                            if not os.path.isdir(folder):
                                print('Make a new folder: ', folder)
                                mkdir_p(folder)                           
                            for k in range(len(fake_imgs2)):
                                im = fake_imgs2[k][j].data.cpu().numpy()
                            # [-1, 1] --> [0, 255]
                                im = (im + 1.0) * 127.5
                                im = im.astype(np.uint8)
                                im = np.transpose(im, (1, 2, 0))
                                im = Image.fromarray(im)
                                fullpath = '%s_s%d_g1_n%d.png' % (s_tmp, dl,nn)
                                im.save(fullpath)
                ppp=ppp+1        

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G2_DCGAN_1()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM

                with torch.no_grad():
                    captions = Variable(torch.from_numpy(captions))
                    cap_lens = Variable(torch.from_numpy(cap_lens))

                    captions = captions.cuda()
                    cap_lens = cap_lens.cuda()
                for nn in range(5):
                    for i in range(1):  # 16
                        with torch.no_grad():
                            noise = Variable(torch.FloatTensor(batch_size, nz))
                            noise = noise.cuda()
                        #######################################################
                        # (1) Extract text embeddings
                        ######################################################
                        hidden = text_encoder.init_hidden(batch_size)
                        # words_embs: batch_size x nef x seq_len
                        # sent_emb: batch_size x nef
                        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                        mask = (captions == 0)
                        #######################################################
                        # (2) Generate fake images
                        ######################################################
                        noise.data.normal_(0, 1)
                        fake_imgs1,fake_imgs2, attention_maps1,attention_maps2, _, _ = netG(noise,noise, sent_emb, words_embs, mask)
                        # G attention
                        cap_lens_np = cap_lens.cpu().data.numpy()
                        for j in range(batch_size):
                            save_name = '%s/%d_s_%d_1' % (save_dir, i, sorted_indices[j])
                            print(save_name)
                            for k in range(len(fake_imgs1)):
                                im = fake_imgs1[k][j].data.cpu().numpy()
                                im = (im + 1.0) * 127.5
                                im = im.astype(np.uint8)
                                # print('im', im.shape)
                                im = np.transpose(im, (1, 2, 0))
                                # print('im', im.shape)
                                im = Image.fromarray(im)
                                fullpath = '%s_g%d_n%d.png' % (save_name, k,nn)
                                im.save(fullpath)

                            for k in range(len(attention_maps1)):
                                if len(fake_imgs1) > 1:
                                    im = fake_imgs1[k + 1].detach().cpu()
                                else:
                                    im = fake_imgs1[0].detach().cpu()
                                attn_maps = attention_maps1[k]
                                att_sze = attn_maps.size(2)
                                img_set, sentences = \
                                    build_super_images2(im[j].unsqueeze(0),
                                                        captions[j].unsqueeze(0),
                                                        [cap_lens_np[j]], self.ixtoword,
                                                        [attn_maps[j]], att_sze)
                                if img_set is not None:
                                    im = Image.fromarray(img_set)
                                    fullpath = '%s_a%d_n%d.png' % (save_name, k,nn)
                                    im.save(fullpath)
                        for j in range(batch_size):
                            save_name = '%s/%d_s_%d_2' % (save_dir, i, sorted_indices[j])
                            for k in range(len(fake_imgs2)):
                                im = fake_imgs2[k][j].data.cpu().numpy()
                                im = (im + 1.0) * 127.5
                                im = im.astype(np.uint8)
                                # print('im', im.shape)
                                im = np.transpose(im, (1, 2, 0))
                                # print('im', im.shape)
                                im = Image.fromarray(im)
                                fullpath = '%s_g%d_n%d.png' % (save_name, k,nn)
                                im.save(fullpath)

                            for k in range(len(attention_maps2)):
                                if len(fake_imgs2) > 1:
                                    im = fake_imgs2[k + 1].detach().cpu()
                                else:
                                    im = fake_imgs2[0].detach().cpu()
                                attn_maps = attention_maps2[k]
                                att_sze = attn_maps.size(2)
                                img_set, sentences = \
                                    build_super_images2(im[j].unsqueeze(0),
                                                        captions[j].unsqueeze(0),
                                                        [cap_lens_np[j]], self.ixtoword,
                                                        [attn_maps[j]], att_sze)
                                if img_set is not None:
                                    im = Image.fromarray(img_set)
                                    fullpath = '%s_a%d_n%d.png' % (save_name, k,nn)
                                    im.save(fullpath)
