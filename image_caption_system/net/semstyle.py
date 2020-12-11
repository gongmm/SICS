import torch
import torch.nn as nn
import os
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .text_processing import untokenize
from . import seq2seq_pytorch as s2s

cuda = False
device = 0

model_path = "/Users/wangyawen/PycharmProjects/SICS/image_caption_system/net/models/"
test_model_fname = "img_to_txt_state.tar"

BATCH_SIZE = 128


class NopModule(torch.nn.Module):
    def __init__(self):
        super(NopModule, self).__init__()

    def forward(self, input):
        return input


def get_cnn():
    # inception = models.inception_v3(pretrained=True, aux_logits=False)
    inception = models.inception_v3(pretrained=True)
    inception.fc = NopModule()
    if cuda:
        inception = inception.to(device=device)
    inception.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])
    return inception, trans


def has_image_ext(path):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    ext = os.path.splitext(path)[1]
    if ext.lower() in IMG_EXTENSIONS:
        return True
    return False


def list_image_folder(root):
    images = []
    dir = os.path.expanduser(root)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if os.path.isdir(d):
            continue
        if has_image_ext(d):
            images.append(d)
    return images


def safe_pil_loader(path, from_memory=False):
    try:
        if from_memory:
            img = Image.open(path)
            res = img.convert('RGB')
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
    except Exception as e:
        res = Image.new('RGB', (299, 299), color=0)
        print(e)
    return res


class ImageTestFolder(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.loader = safe_pil_loader
        self.transform = transform

        self.samples = list_image_folder(root)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)


# load images provided across the network
class ImageNetLoader(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.loader = safe_pil_loader
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loader(self.images[index], from_memory=True)
        sample = self.transform(sample)
        return sample, ""

    def __len__(self):
        return len(self.images)


def get_image_reader(dirpath, transform, batch_size, workers=4):
    image_reader = torch.utils.data.DataLoader(
        ImageTestFolder(dirpath, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return image_reader


class ImgEmb(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.mlp = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        res = self.relu(self.mlp(input))
        return res


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_bias=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.5)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.5)
        self.mlp = nn.Linear(hidden_size, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_in):
        emb = self.embedding(input)
        out, hidden = self.gru(self.emb_drop(emb), hidden_in)
        out = self.mlp(self.gru_drop(out))
        out = self.logsoftmax(out)
        return out, hidden


def build_model(dec_vocab_size, dec_bias=None, img_feat_size=2048,
                hid_size=512, loaded_state=None):
    enc = ImgEmb(img_feat_size, hid_size)
    dec = Decoder(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec


def generate(enc, dec, feats, L=20):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        hid_enc = enc(feats).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.zeros(feats.shape[0], L + 1, dtype=torch.long)
        if cuda:
            dec_tensor = dec_tensor.to(device=device)
        last_enc = hid_enc
        for i in range(L):
            out_dec, hid_dec = dec.forward(dec_tensor[:, i].unsqueeze(1), last_enc)
            chosen = torch.argmax(out_dec[:, 0], dim=1)
            dec_tensor[:, i + 1] = chosen
            last_enc = hid_dec

    return dec_tensor.data.cpu().numpy()


def setup_test(with_cnn=False):
    print("loading CNN")
    cnn, trans = get_cnn()
    print("loading image to text model")
    if not cuda:
        loaded_state = torch.load(model_path + test_model_fname,
                                  map_location='cpu')
    else:
        loaded_state = torch.load(model_path + test_model_fname)
    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    print("building image to text model")
    enc, dec = build_model(dec_vocab_size, loaded_state=loaded_state)

    s2s.cuda = cuda
    s2s_data = s2s.setup_test()
    return {'cnn': cnn, 'trans': trans, 'enc': enc, 'dec': dec,
            'loaded_state': loaded_state, 's2s_data': s2s_data}


class TestIterator:
    def __init__(self, feats, text, bs=BATCH_SIZE):
        self.feats = feats
        self.text = text
        self.bs = bs
        self.num_batch = feats.shape[0] // bs
        if feats.shape[0] % bs != 0:
            self.num_batch += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.num_batch:
            raise StopIteration()

        s = self.i * self.bs
        e = min((self.i + 1) * self.bs, self.feats.shape[0])
        self.i += 1
        return self.feats[s:e], self.text[s:e]


def test(setup_data, test_images=None):
    enc = setup_data['enc']
    dec = setup_data['dec']
    cnn = setup_data['cnn']
    trans = setup_data['trans']
    loaded_state = setup_data['loaded_state']
    s2s_data = setup_data['s2s_data']

    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    img_reader = DataLoader(
        ImageNetLoader(test_images, trans),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=True)

    all_text = []
    for input, text_data in img_reader:
        if cuda:
            input = input.to(device=device)
        with torch.no_grad():
            batch_feats_tensor = cnn(input)

        dec_tensor = generate(enc, dec, batch_feats_tensor)

        untok = []
        for i in range(dec_tensor.shape[0]):
            untok.append(untokenize(dec_tensor[i],
                                    loaded_state['dec_idx_to_word'],
                                    to_text=False))

        text = s2s.test(s2s_data, untok)

        all_text.extend(text)
    return all_text


def main():
    test_dir = "/data/wangziang/projects/stylenet/data/flickr7k_images"
    print("setup test")
    r = setup_test()
    print("test image")
    for image in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image)
        result = test(r, test_images=[image_path])
        print(image_path + " result: " + str(result))


if __name__ == "__main__":
    main()
