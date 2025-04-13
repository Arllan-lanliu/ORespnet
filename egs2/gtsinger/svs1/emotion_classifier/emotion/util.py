import torch
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, loss_history, filename="./files/checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history  
    }
    torch.save(checkpoint, filename)
    
    print(f"[Util] Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, filename="./files/checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_history = checkpoint.get('loss_history', [])  # 读取 loss 记录
    print(f"[Util] Checkpoint loaded, resuming from epoch {start_epoch}")
    return start_epoch, loss_history


def draw_loss(losses_list, epoch, savepath="./files/training_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), losses_list, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(savepath)
    print("[Util] Training Loss Curve saved ") 
