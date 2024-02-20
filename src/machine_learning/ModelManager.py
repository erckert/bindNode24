import torch


def save_classifier_torch(classifier, model_path):
    """Save pre-trained model"""
    torch.save(classifier, model_path)


def load_classifier_torch(model_path):
    """ Load pre-saved model """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    classifier = torch.load(model_path, map_location=device)
    return classifier
