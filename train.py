import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.NLINet import *
import tqdm
import json
import argparse
from torch.utils.tensorboard import SummaryWriter


from data_preprocessing import data_preprocessing

def train(model, vocab, train_loader, args, val_loader = None, test_loader = None, checkpoint_path = None, device = 'cuda', lr = 0.1, max_num_epochs = 20):
    # saving code
    os.makedirs(checkpoint_path, exist_ok=True)
    print('checkpoint directory created')
    save_path = os.path.join(checkpoint_path, args.encoder_name + ".pt")

    print(f'Starting training of {args.encoder_name}')

    writer = SummaryWriter(os.path.join(args.log_dir, args.encoder_name))
    val_losses, val_scores = [], []
    train_losses, train_scores = [], []
    top_val_acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    
    # checkpoint_path = f"{model.encoder_name}.pt"
    for epoch in range(max_num_epochs):
        if lr<1e-5:
            print("Minimum LR breached, training stopped")
            break
        epoch_loss = 0.0
       
        model.train()
        # print('start training, model device is ', model.device)
        true_preds, count = 0., 0
        '''
        1. load data
        2. get preds
        3. get loss
        4. backprop
        5. update optimizer
            5.1 lr update
        '''
        for sentences in train_loader:
            
            # take the unpadded sentence lengths
            premise_lengths = torch.tensor([torch.sum(x != 1) for x in sentences.premise]).to(device)
            hypo_lengths = torch.tensor([torch.sum(x != 1) for x in sentences.hypothesis]).to(device)
            
            # take the numerical sentence translation (no embedding -> handled by the model)
            premise = sentences.premise.to(device)
            hypo = sentences.hypothesis.to(device)


            # get preds 
            preds = model((premise, premise_lengths), (hypo, hypo_lengths))
            # the labels run from 1 - 3 from torchtext, - 1 to adjust to [0, C)
            labels = (sentences.label - 1).to(device)
            
            # get loss & backprop
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
    

            # record some statistics during training batches
            true_preds += (preds.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]

        train_acc = true_preds / count
        train_scores.append(train_acc)
        epoch_loss = epoch_loss / len(train_loader)

        train_scores.append(train_acc)
        train_losses.append(epoch_loss)

        writer.add_scalar('Loss/Train',
                          epoch_loss,
                          global_step = epoch + 1)
        writer.add_scalar('Accuracy/Train',
                          train_acc,
                          global_step = epoch + 1)

        # lr decay of factor 0.99 after every epoch
        lr = lr * 0.99
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print(f"learning rate decay at end of epoch {epoch}, lr is now {optimizer.param_groups[0]['lr']}")

        ##############
        # Validation #
        ##############
        val_acc, val_loss = eval_model(model, val_loader, device)
        val_scores.append(val_acc)
        val_losses.append(val_loss)


        writer.add_scalar('Accuracy/Val',
                          val_acc,
                          global_step = epoch + 1)
        
        writer.add_scalar('Loss/Val',
                          val_loss,
                          global_step = epoch + 1)

        # val_scores.append(val_acc)
        print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%")

        if val_acc > top_val_acc:
            print("\t   (New best performance, saving model...)")
            top_val_acc = val_acc
            # saving best model
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
            }, save_path)
            print(f"Saved model checkpoint at epoch {epoch} with validation accuracy {val_acc}")

        else:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / args.lr_shrink
            print(f"learning rate has been updated by a division of 5, new lr is {optimizer.param_groups[0]['lr']}")
    

    model = NLINet(encoder_name=args.encoder_name, vocab = vocab)
    model.load_state_dict(torch.load(save_path)["model_state_dict"])
    print("Succesfully loaded model..")
    model.to(device)
    test_acc, _ = eval_model(model, test_loader, model.device)
    results = {"test_acc": test_acc, "val_scores": val_scores, "train_losses": train_losses, "train_scores": train_scores}

    print(f'TRAINING DONE... FINAL TEST ACCURACY IS {test_acc*100.0:05.2f}')
    filename = f"{args.encoder_name}_results.json"
    filepath = os.path.join(checkpoint_path, filename)
    with open(filepath, "w") as f:
        json.dump(results, f)
    writer.add_scalar('Accuracy/Test', test_acc)
    writer.close()


def eval_model(model, data_loader, device):
    model.eval() # Set model to eval mode
    true_preds, count = 0., 0.
    val_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad(): # Deactivate gradients for the following code
        for sentences in data_loader:
            
            # take the unpadded sentence lengths
            premise_lengths = torch.tensor([torch.sum(x != 1) for x in sentences.premise]).to(device)
            hypo_lengths = torch.tensor([torch.sum(x != 1) for x in sentences.hypothesis]).to(device)
            
            # take the numerical sentence translation (no embedding -> handled by the model)
            premise = sentences.premise.to(device)
            hypo = sentences.hypothesis.to(device)


            # get preds 
            preds = model((premise, premise_lengths), (hypo, hypo_lengths))
            # the labels run from 1 - 3 from torchtext, - 1 to adjust to [0, C)
            labels = (sentences.label - 1).to(device)

            # get loss
            loss = loss_fn(preds, labels)
            val_loss += loss.item()


            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (preds.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
    eval_acc = true_preds / count
    eval_loss = val_loss/count
    return eval_acc, eval_loss

               

           





if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model
    parser.add_argument('--encoder_name', type=str, default='uniLSTM',
                        help = 'see list of encoders')
    
    # training
    parser.add_argument("--lr", type=float, default=0.1, help="lr for sgd")
    parser.add_argument("--lr_shrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--lr_decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    # miscellaneuos
    parser.add_argument("--log_dir", type=str, default='tf_logs/', help="logging directory")
    
    args = parser.parse_args()

    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vocab, train_loader, val_loader, test_loader = data_preprocessing(batch_size=64)      
    model = NLINet(encoder_name=args.encoder_name, vocab = vocab)
    SAVE_PATH = f"./checkpoints"
    train(model=model, vocab=vocab, train_loader=train_loader, args=args, val_loader=val_loader, test_loader=test_loader, checkpoint_path = SAVE_PATH)

    # load model and return test accuracy