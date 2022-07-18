import torch
from torch.utils.data import DataLoader
import os
import math
from config import *
from model import NvNet
from transforms import train_transforms, val_transforms
from dataset import BraTSDataSet
from criteria import CombinedLoss



# Train DataLoader Definition
train_dataset = BraTSDataSet(img_root=train_img_root, label_root=train_label_root, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=os.cpu_count())

# Val DataLoader Definition
val_dataset = BraTSDataSet(img_root=val_img_root, label_root=val_label_root, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

# Network Instantiation
net = NvNet(inChans, input_shape, seg_outChans, activation, normalizaiton, VAE_enable, mode='trilinear')
if torch.cuda.is_available(): net = net.cuda()

# Defining Loss Function and Optimizer
criterion = CombinedLoss(k1=0.1, k2=0.1)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

best_loss = -math.inf
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True



for epoch in range(0, epochs):

    # Train Model
    print('\n\n\nEpoch: {}\n<Train>'.format(epoch))
    net.train(True)
    loss = 0
    lr = lr * (0.5 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    torch.set_grad_enabled(True)
    for idx, (img, label) in enumerate(train_loader):
        if torch.cuda.is_available():
          img, label = img.cuda(), label.cuda()
        pred = net(img)
        seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
        batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss += float(batch_loss)
    log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
    print(log_msg)


    # Validate Model
    print('\n\n<Validation>')
    net.eval()
    for module in net.module.modules():
        if isinstance(module, torch.nn.modules.Dropout2d):
            module.train(True)
        elif isinstance(module, torch.nn.modules.Dropout):
            module.train(True)
        else:
            pass
    loss = 0
    torch.set_grad_enabled(False)
    for idx, (img, label) in enumerate(val_loader):
      if torch.cuda.is_available():
        img, label = img.cuda(), label.cuda()
        pred = net(img)
        seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
        batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
        loss += float(batch_loss)
    log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
    print(log_msg)

    # Save Model
    if loss <= best_loss:
        torch.save(os.path.join(checkpoint_path, f'epoch:{epoch}_loss{loss}.tar'))
        best_loss = loss
        print("Saving...")