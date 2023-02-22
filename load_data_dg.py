from load_data import CATEGORIES, PACSDatasetBaseline, PACSDatasetTuple, read_lines
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import json

DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']

# Application of transforms
class PACSDatasetTupleDg(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example, y = self.examples[index]
        x = self.transform(Image.open(example[0]).convert('RGB'))
        return (x, example[1], example[2]), y

# Read the paths of images and prepare a dictionary category id - [(image paths, domain ids)]
def read_lines_dg(data_path, domain_name, dom_idx = None):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [(image_path, dom_idx)]
        else:
            examples[category_idx].append((image_path, dom_idx))
    return examples

# Read the paths of images and prepare a dictionary category id - [(image paths, domain ids, descriptions)]
def read_lines_clip_dg(data_path, domain_name, dom_idx, with_desc):
    examples = {}
    
    if with_desc:
        with open(f'{data_path}/descriptions/{domain_name}_descriptions.txt') as f:
            for line in f.readlines(): 
                dict_tmp = json.loads(line.strip())
                path = dict_tmp['image_name'].split('/')
                category_name = path[1]
                category_idx = CATEGORIES[category_name]
                image_name = path[2]
                image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
                description = dict_tmp['descriptions']
                if category_idx not in examples.keys():
                    examples[category_idx] = [(image_path, dom_idx, description)]
                else:
                    examples[category_idx].append((image_path, dom_idx, description))
    else:
        with open(f'{data_path}/descriptions/{domain_name}_without_descriptions.txt') as f:
            for line in f.readlines(): 
                line = line.strip().split()[0].split('/')
                category_name = line[3]
                category_idx = CATEGORIES[category_name]
                image_name = line[4]
                image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
                if category_idx not in examples.keys():
                    examples[category_idx] = [(image_path, dom_idx)]
                else:
                    examples[category_idx].append((image_path, dom_idx))

    return examples

def build_splits_baseline_dg(opt):
    
    target_domain = opt['target_domain']
    source_domains = list(filter(lambda dom : dom != target_domain, DOMAINS))

    # Acquiring data from all the domains
    source_examples = {}
    for dom in source_domains:
        tmp_examples = read_lines(opt['data_path'], dom)
        for key in tmp_examples.keys():
            if key in source_examples:
                source_examples[key].extend(tmp_examples[key])
            else:
                source_examples[key] = tmp_examples[key]

    target_examples = read_lines(opt['data_path'], target_domain)

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        for i, example in enumerate(examples_list):
            if i % 5 != 0:
                train_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
            else:
                val_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

    
def build_splits_domain_disentangle_dg(opt):

    target_domain = opt['target_domain']
    source_domains = list(filter(lambda dom : dom != target_domain, DOMAINS))

    # Acquiring data from all the domains
    source_examples = {}
    # Generate domain labels enumerating source_domains
    for dom_idx, dom in enumerate(source_domains):
        tmp_examples = read_lines_dg(opt['data_path'], dom, dom_idx)
        for key in tmp_examples.keys():
            if key in source_examples:
                source_examples[key].extend(tmp_examples[key])
            else:
                source_examples[key] = tmp_examples[key]

    target_examples = read_lines(opt['data_path'], target_domain)

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        for i, example in enumerate(examples_list):
            if i % 5 != 0:
                train_examples.append([example, category_idx]) # each pair is [(path_to_img, domain_idx), class_label]
            else:
                val_examples.append([example[0], category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetTuple(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

def build_splits_clip_disentangle_dg(opt):
    
    target_domain = opt['target_domain']
    source_domains = list(filter(lambda dom : dom != target_domain, DOMAINS))

    # Acquiring data from all the domains, separately for images with and without descriptions
    source_examples = {}
    source_examples_descriptions = {}
    # Generate domain labels enumerating source_domains
    for dom_idx, dom in enumerate(source_domains):
        tmp_examples = read_lines_clip_dg(opt['data_path'], dom, dom_idx, 0)
        for key in tmp_examples.keys():
            if key in source_examples:
                source_examples[key].extend(tmp_examples[key])
            else:
                source_examples[key] = tmp_examples[key]
        
        tmp_examples_descriptions = read_lines_clip_dg(opt['data_path'], dom, dom_idx, 1)
        for key in tmp_examples_descriptions.keys():
            if key in source_examples_descriptions:
                source_examples_descriptions[key].extend(tmp_examples_descriptions[key])
            else:
                source_examples_descriptions[key] = tmp_examples_descriptions[key]

    target_examples = read_lines(opt['data_path'], target_domain)

    train_examples = []
    descriptions_train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        for i, example in enumerate(examples_list):
            if i % 5 != 0:
                train_examples.append([example, category_idx]) # each pair is [(path_to_img, domain_idx, description), class_label]
            else:
                val_examples.append([example[0], category_idx]) # each pair is [path_to_img, class_label]
    
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx]) # each pair is [path_to_img, class_label]

    #With descriptions
    for category_idx, examples_list in source_examples_descriptions.items():
        for i, example in enumerate(examples_list):
            descriptions_train_examples.append([example, category_idx]) # each pair is [(path_to_img, domain_idx, description), class_label]
    
    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetTuple(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    descriptions_train_loader = DataLoader(PACSDatasetTupleDg(descriptions_train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)

    return train_loader, val_loader, test_loader, descriptions_train_loader