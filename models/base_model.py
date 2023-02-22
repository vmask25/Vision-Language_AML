import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x_r = x.squeeze()
        if len(x_r.size()) < 2:
          x_r = x_r.unsqueeze(0)
        return x_r

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier_dg = nn.Linear(512, 3)
        self.domain_classifier = nn.Linear(512, 2)
        self.category_classifier = nn.Linear(512, 7)

        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, x, is_train, dg=False):
        # x = feature, y classification result
        # c = category, d = domain
        # a = adversarial

        x = self.feature_extractor(x)
        c_x = self.category_encoder(x)
        d_x = self.domain_encoder(x)

        # is_train = discriminate between train and evaluation
        # dg = specify if domain generalization flag is set
        if is_train:
            fg = c_x + d_x
            r_x = self.reconstructor(fg)
            c_y = self.category_classifier(c_x)
            if not dg:
                d_y = self.domain_classifier(d_x)
            else:
                d_y = self.domain_classifier_dg(d_x)
            # Giving domain encoded features to category classifier (adversarial)
            a_c_y = self.category_classifier(d_x)
            # Giving category encoded features to domain classifier (adversarial)
            a_d_y = self.domain_classifier(c_x)
            return x, r_x, c_y, d_y, a_c_y, a_d_y
        else:
            c_y = self.category_classifier(c_x)
            return c_y

class CLIPDisentangleModel(nn.Module):
    def __init__(self):
        super(CLIPDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier_dg = nn.Linear(512, 3)
        self.domain_classifier = nn.Linear(512, 2)
        self.category_classifier = nn.Linear(512, 7)

        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, x, is_train, dg=False):
        # x = feature, y classification result
        # c = category, d = domain
        # a = adversarial

        x = self.feature_extractor(x)
        c_x = self.category_encoder(x)
        d_x = self.domain_encoder(x)

        # is_train = discriminate between train and evaluation
        # dg = specify if domain generalization flag is set
        if is_train:
            fg = c_x + d_x
            r_x = self.reconstructor(fg)
            c_y = self.category_classifier(c_x)
            if not dg:
                d_y = self.domain_classifier(d_x)
            else:
                d_y = self.domain_classifier_dg(d_x)
            # Giving domain encoded features to category classifier (adversarial)
            a_c_y = self.category_classifier(d_x)
            # Giving category encoded features to domain classifier (adversarial)
            a_d_y = self.domain_classifier(c_x)
            return x, r_x, c_y, d_y, a_c_y, a_d_y, d_x
        else:
            c_y = self.category_classifier(c_x)
            return c_y
