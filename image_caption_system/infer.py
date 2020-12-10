import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import sys
import os
import pickle

rootpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(rootpath, 'net')
sys.path.append(path)
from .net.stylenet import EncoderCNN, FactoredLSTM


def get_style_image_caption(image_path, vocab_path, encoder_model_path, factual_decoder_model_path, style_decoder_model_path):
    # 加载模型
    vocab, encoder, factual_decoder = load_stylenet_model(vocab_path, encoder_model_path, factual_decoder_model_path)
    vocab, encoder, style_decoder = load_stylenet_model(vocab_path, encoder_model_path, style_decoder_model_path)
    print("process image: ", image_path)
    output = get_image_caption(image_path, encoder, factual_decoder, style_decoder)
    caption = [vocab.i2w[x] for x in output]
    return caption


def get_image_caption(input_image, encoder, factual_decoder, style_decoder):
    """
    根据人脸，给概率
    :param input_image:
    :return:
    """
    transforms_set = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    example = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    example = Image.fromarray(example)
    example = transforms_set(example)
    with torch.no_grad():
        feature = encoder(input_image.unsqueeze(0))
        caption = factual_decoder.sample(feature, mode="factual")
        output = style_decoder(caption.unsqueeze(0), feature, mode="romantic")
    return output


def load_stylenet_model(vocab_path, encoder_model_path, decoder_model_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    encoder = EncoderCNN(300, device)
    decoder = FactoredLSTM(300, 512, 512, len(vocab))
    print("loading encoder state dict %s" % encoder_model_path)
    try:
        encoder_checkpoint = torch.load(encoder_model_path, map_location="cpu")
        encoder.load_state_dict(encoder_checkpoint, strict=False)
    except Exception as e:
        print("error loading %s" % encoder_model_path)
        print(e)
    print("loading encoder state dict %s" % decoder_model_path)
    try:
        decoder_checkpoint = torch.load(decoder_model_path, map_location="cpu")
        decoder.load_state_dict(decoder_checkpoint, strict=False)
    except Exception as e:
        print("error loading %s" % decoder_model_path)
        print(e)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    print("load model ready")
    return vocab, encoder, decoder


if __name__ == '__main__':
    pass
