
import torch

class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc = torch.nn.Linear(input_size, num_classes).to(self._device) #256 -> 2
    
    
    def get_device(self):
        return self._device

    def forward(self, x): #x=[B,256]

        return self.fc(x)