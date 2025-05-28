import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def Tokenize(sent):
    url = "microsoft/BiomedVLP-CXR-BERT-specialized"
    tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
    tokens = tokenizer(
        sent,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=112,
    )
    x_len = len([t for t in tokens["input_ids"][0] if t != 0])
    return tokens, x_len

def generate_prompt(subject):
    status = {
        "stable": ['constant', 'stable', 'unchanged'],
        "worsen": ['bigger', 'developing', 'enlarged', 'enlarging', 'greater', 'growing', 'increased', 'increasing', 
                   'larger', 'new', 'progressing', 'progressive', 'worse', 'worsened', 'worsening'],
        "improving": ['better', 'cleared', 'decreased', 'decreasing', 'im-proved', 'improving', 'reduced', 
                      'resolved', 're-solving', 'smaller']
    }
    
    prompt = "Finding is that the " + subject + " progression is "

    caption_ids = []
    attention_mask = []
    token_type_ids = []
    for category, terms in status.items():
        for term in terms:
            sent = prompt + term + '.'
            sent_token, _ = Tokenize(sent)  # I'm assuming you have a Tokenize function defined elsewhere in your code
            caption_ids.append(sent_token["input_ids"])
            attention_mask.append(sent_token["attention_mask"])
            token_type_ids.append(sent_token["token_type_ids"])

    caption_ids = torch.stack(caption_ids).squeeze()  
    attention_mask = torch.stack(attention_mask).squeeze()   
    token_type_ids = torch.stack(token_type_ids).squeeze() 

    return caption_ids, attention_mask, token_type_ids, len(status["stable"]), len(status["worsen"]), len(status["improving"])



def classify_by_similarity(img_feat: torch.Tensor, text_feats: torch.Tensor):
    """
    Classify an image based on the similarity of its feature to a set of text features.
    
    Parameters
    ----------
    img_feat : torch.Tensor
        The feature of the image, expected shape is (1, 128).
    text_feats : torch.Tensor
        The features of the text, expected shape is (3, 128).

    Returns
    -------
    int
        The index of the text feature that is most similar to the image feature.
    """
    # Calculate the cosine similarity of the image feature with each text feature
    similarity_scores = F.cosine_similarity(img_feat, text_feats, dim=-1)
    
    # Get the index of the most similar text feature
    classification_result = torch.argmax(similarity_scores)

    return classification_result.item()