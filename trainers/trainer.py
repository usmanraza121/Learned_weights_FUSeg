import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import pandas as pd


def evaluate(model, device, test_loader, class_names, debug=True):

    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                total_pred[class_names[label]] += 1

    accuracy = correct / total

    if debug:
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return accuracy
# from train import image_loader_train
# image_type = image_loader_train.image_type

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, ckpt_folder, num_epochs=25):
    start = time.time()
    # print('image---', image_type)
    # Create a folder for checkpoints
    best_model_file_name = 'best_model_' + model.name + '_params.pt' 
    best_model_path = os.path.join(ckpt_folder, best_model_file_name)

    torch.save(model.state_dict(), best_model_path)
    best_acc = 0.0
    
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 75)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()   
                
            running_loss = 0.0
            running_true = 0
            running_acc = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # test_v = torch.clone(outputs).cpu().detach().numpy()
                    # print(np.min(test_v), np.max(test_v), np.mean(test_v))
                    #_, preds = torch.max(outputs, 1)
                    #loss, acc = criterion(preds, labels)
                    loss, acc = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += acc.item() * inputs.size(0)
                # running_true += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            # epoch_acc = running_true.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            else:
                valid_loss_history.append(epoch_loss)
                valid_acc_history.append(epoch_acc)

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)

            epoch_time = time.time() - start
            print(f'{phase} Epoch completed in: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
        
        print()

    time_elapsed = time.time() - start
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid accuracy: {best_acc:4f}')

    # Load weights for the best model
    model.load_state_dict(torch.load(best_model_path))
    # model.load_state_dict(torch.load(best_model_path, weights_only=True))
    results_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'valid_loss': valid_loss_history,
        'valid_acc': valid_acc_history})


    save_training_results(train_loss_history, train_acc_history, valid_loss_history, 
                                            valid_acc_history, num_epochs, ckpt_folder, model.name) 
    
    return model

# save_training_results(train_loss_history, train_acc_history, valid_loss_history, 
#                                           valid_acc_history, num_epochs, ckpt_folder, model.name)

def save_training_results(train_loss_history, train_acc_history, valid_loss_history, 
                                            valid_acc_history, num_epochs, ckpt_folder, model_name):

    results_df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_loss_history,
        'train_acc': train_acc_history,
        'valid_loss': valid_loss_history,
        'valid_acc': valid_acc_history})

    plots_dir = os.path.join(ckpt_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    results_csv_path = os.path.join(plots_dir, f'training_results_{model_name}.csv')
    # results_csv_path = os.path.join(plots_dir, 'training_results_' + model.name + '.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f'Training results saved to {results_csv_path}')

# def train_model2(model, criterion, optimizer, scheduler, device, dataloaders, 
#                                     dataset_sizes, ckpt_folder, num_epochs=25):

    
#     start = time.time()
#     model_name = getattr(model, 'name', 'unnamed_model')
#     best_model_path = os.path.join(ckpt_folder, f'best_model_{model_name}_{image_type}_params.pt')

#     best_acc = 0.0
#     train_loss_history = []
#     train_acc_history = []
#     valid_loss_history = []
#     valid_acc_history = []

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 75)
#         epoch_start = time.time()

#         for phase in ['train', 'valid']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_acc = 0.0

#             for inputs, labels in dataloaders[phase]:
#                 inputs, labels = inputs.to(device), labels.to(device).squeeze(1)
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss, acc = criterion(outputs, labels)
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_acc += acc.item() * inputs.size(0)

#             if phase == 'train':
#                 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(running_loss / dataset_sizes[phase])
#                 else:
#                     scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_acc / dataset_sizes[phase]
#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             if phase == 'valid' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scheduler_state_dict': scheduler.state_dict(),
#                     'best_acc': best_acc,
#                     'epoch': epoch,
#                 }, best_model_path)

#         epoch_time = time.time() - epoch_start
#         print(f'Epoch completed in: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
#         print()

#     time_elapsed = time.time() - start
#     print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best valid accuracy: {best_acc:4f}')

#     checkpoint = torch.load(best_model_path)
#     model.load_state_dict(checkpoint['model_state_dict'])

#     return model, best_acc
