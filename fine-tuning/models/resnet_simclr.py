import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.resnet_wider import resnet50x1, resnet50x2, resnet50x4



class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, num_class,frozen):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet50" : resnet50x1()}


        resnet = self._get_basemodel(base_model)

        # sd = torch.load('./models/model.pth', map_location='cpu')
        # resnet.load_state_dict(sd['state_dict'])

        # print("pretrained simclr loaded")


        # for child  in resnet.children():

        #     for param in list(child.parameters()):

        #         print(child,param[3][0])

        #         break
        #     break



        # if base_model=='custom':
        #     num_ftrs = resnet.layer.out_features
        #     self.features=resnet

    
        num_ftrs = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])


        if frozen:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

        
        #This is special.. SimCLR have two hidden layers , win the inferece they use only the 1st hidden layer
        # projection MLP
        self.out = nn.Linear(num_ftrs, num_class)
     

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):


        h = self.features(x)

        h = h.squeeze()

        x = self.out(h)
   
    
        return x
