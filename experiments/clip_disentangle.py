import torch
from models.base_model import CLIPDisentangleModel
import clip

class CLIPDisentangleExperiment: # See point 4. of the project
    
    def __init__(self, opt):
         # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = CLIPDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup clip model
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        # Entropy loss following Shannon Entropy definition
        self.entropyLoss = lambda outputs : -torch.mean(torch.sum(self.logSoftmax(outputs), dim=1))
        self.mseloss = torch.nn.MSELoss()

        # Setting weights for loss functions
        # weights[0] => category classifier loss
        # weights[1] => domain classifier loss
        # weights[2] => reconstruction loss
        # weights[3] => clip loss
        self.weights = [10,5,1,1]

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, domain, desc): 
        # domain == 0 -> source
        # domain == 1 -> target
        # desc == 0 -> without descriptions
        # desc == 1 -> with descriptions
        
        images = []
        labels = []

        self.optimizer.zero_grad()

        if not self.opt['dom_gen']:
            if desc == 1:
                descriptions = []
                if domain == 0:
                    examples, labels = data
                    images, descriptions = examples
                    descriptions = list(descriptions[0])
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Network outputs
                    features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs, source_domain_features = self.model(images, True)
                    source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
                    # Computing cross entropy loss with tensor of labels 0 (source batch)
                    source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, torch.zeros(source_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                    reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                    # Adversarial losses: 
                    # category features -> domain classifier
                    source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
                    # domain features -> category classifier
                    source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
                    # Tokenizing and passing descriptions to the clip text encoder
                    tokenized_text = clip.tokenize(descriptions).to(self.device)
                    text_features = self.clip_model.encode_text(tokenized_text)
                    clip_loss = self.weights[3]*self.mseloss(text_features, source_domain_features)
                    # Total loss summing all the computed partial losses
                    total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss + clip_loss
                else:
                    examples, _ = data
                    images, descriptions = examples
                    descriptions = list(descriptions[0])
                    images = images.to(self.device)
                    # Network outputs
                    features, rec_features, _ , target_dom_outputs, target_adv_objC_outputs, target_adv_domC_outputs, target_domain_features = self.model(images, True)
                    # Computing cross entropy loss with tensor of labels 1 (target batch)
                    target_dom_loss = self.weights[0]*self.crossEntropyLoss(target_dom_outputs, torch.ones(target_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                    reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                    # Adversarial losses: 
                    # category features -> domain classifier
                    target_adv_domC_loss =  self.weights[0]*self.opt["alpha"]*self.entropyLoss(target_adv_domC_outputs)
                    # domain features -> category classifier
                    target_adv_objC_loss = self.weights[1]*self.opt['alpha']*self.entropyLoss(target_adv_objC_outputs)
                    # Tokenizing and passing descriptions to the clip text encoder
                    tokenized_text = clip.tokenize(descriptions).to(self.device)
                    text_features = self.clip_model.encode_text(tokenized_text)
                    clip_loss = self.weights[3]*self.mseloss(text_features, target_domain_features)
                    # Total loss summing all the computed partial losses
                    total_loss = (target_dom_loss + target_adv_domC_loss) + target_adv_objC_loss + reconstruction_loss + clip_loss
            else:
                if domain == 0:
                    images, labels = data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Network outputs
                    features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs, _ = self.model(images, True)
                    source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
                    # Computing cross entropy loss with tensor of labels 0 (source batch)
                    source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, torch.zeros(source_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                    reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                    # Adversarial losses: 
                    # category features -> domain classifier
                    source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
                    # domain features -> category classifier
                    source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
                    # Total loss summing all the computed partial losses
                    total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss
                else:
                    images, _ = data
                    images = images.to(self.device)
                    # Network outputs
                    features, rec_features, _ , target_dom_outputs, target_adv_objC_outputs, target_adv_domC_outputs, _ = self.model(images, True)
                    # Computing cross entropy loss with tensor of labels 1 (target batch)
                    target_dom_loss = self.weights[0]*self.crossEntropyLoss(target_dom_outputs, torch.ones(target_dom_outputs.size()[0], dtype = torch.long).to(self.device))
                    reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                    # Adversarial losses: 
                    # category features -> domain classifier
                    target_adv_domC_loss =  self.weights[0]*self.opt["alpha"]*self.entropyLoss(target_adv_domC_outputs)
                    # domain features -> category classifier
                    target_adv_objC_loss = self.weights[1]*self.opt['alpha']*self.entropyLoss(target_adv_objC_outputs)
                    # Total loss summing all the computed partial losses
                    total_loss = (target_dom_loss + target_adv_domC_loss) + target_adv_objC_loss + reconstruction_loss
        else:
            if desc == 1:
                descriptions = []
                examples, labels = data
                images, dom_labels, descriptions = examples
                descriptions = list(descriptions[0])
                images = images.to(self.device)
                dom_labels = dom_labels.to(self.device)
                labels = labels.to(self.device)
                # Network outputs
                features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs, source_domain_features = self.model(images, True, True)
                source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
                # Computing cross entropy loss with tensor of domain labels
                source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, dom_labels)
                reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                # Adversarial losses: 
                # category features -> domain classifier
                source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
                # domain features -> category classifier
                source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
                # Tokenizing and passing descriptions to the clip text encoder
                tokenized_text = clip.tokenize(descriptions).to(self.device)
                text_features = self.clip_model.encode_text(tokenized_text)
                clip_loss = self.weights[3]*self.mseloss(text_features, source_domain_features)
                # Total loss summing all the computed partial losses
                total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss + clip_loss
            else:
                examples, labels = data
                images, dom_labels = examples
                images = images.to(self.device)
                dom_labels = dom_labels.to(self.device)
                labels = labels.to(self.device)
                # Network outputs
                features, rec_features, source_class_outputs, source_dom_outputs, source_adv_objC_outputs, source_adv_domC_outputs, _ = self.model(images, True, True)
                source_class_loss = self.weights[0]*self.crossEntropyLoss(source_class_outputs, labels)
                # Computing cross entropy loss with tensor of domain labels
                source_dom_loss = self.weights[1]*self.crossEntropyLoss(source_dom_outputs, dom_labels)
                reconstruction_loss = self.weights[2]*self.mseloss(rec_features, features)
                # Adversarial losses: 
                # category features -> domain classifier
                source_adv_domC_loss = self.weights[0]*self.opt["alpha"]*self.entropyLoss(source_adv_domC_outputs)
                # domain features -> category classifier
                source_adv_objC_loss = self.weights[1]*self.opt["alpha"]*self.entropyLoss(source_adv_objC_outputs)
                # Total loss summing all the computed partial losses
                total_loss = (source_class_loss + source_adv_domC_loss) + (source_dom_loss + source_adv_objC_loss) + reconstruction_loss
        
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, False)
                loss += self.crossEntropyLoss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss