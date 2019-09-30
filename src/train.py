from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from model.trans_NFCM import TransNFCM
from optimizer.radam import RAdam
from feature.data_loader_for_NFCM import FMNISTDataset, loader


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

idx2label = {0:"T-shirt/top",
             1:"Trouser",
             2:"Pullover",
             3:"Dress",
             4:"Coat",
             5:"Sandal",
             6:"Shirt",
             7:"Sneaker",
             8:"Bag",
             9:"Ankle boot"}


def main(args):
    n_relational_embeddings = args.n_class**2
    n_tag_embeddings = args.n_class
    in_ch, out_ch = 1, 128
    model = TransNFCM(in_ch, out_ch, n_tag_embeddings,
                      embedding_dim=128).to(device)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = RAdam(model.parameters(), weight_decay=1e-3)

    train_dataset = FMNISTDataset(n_class=args.n_class, train=True)
    test_dataset = FMNISTDataset(n_class=args.n_class, train=False)

    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    test(args, model, test_loader,
         show_image_on_board=args.show_image_on_board,
         show_all_embedding=args.show_all_embedding)


def train(args, model, optimizer, data_loader):
    model.train()
    for epoch in range(args.epochs):
        for i, (image, cat) in enumerate(data_loader):
            image = image.to(device)
            cat = cat.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            loss = model(image, cat)
            loss.backward()
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                       '{}/model.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
               '{}/model.pth'.format(args.out_dir))


def test(args, model, data_loader,
         show_image_on_board=True, show_all_embedding=False):
    model.eval()
    writer = SummaryWriter()
    weights = []
    images = []
    labels = []
    with torch.no_grad():
        for i, (image, cat) in enumerate(data_loader):
            if i==1000: break
            image = image.to(device)
            cat = cat.to(device)
            labels.append(idx2label[cat.item()])
            images.append(image.squeeze(0).numpy())

            if show_image_on_board:
                if show_all_embedding:
                    embedded_vec = model.predict(x=image, category=cat)
                else:
                    embedded_vec = model.predict(x=image, category=None)
            else:
                embedded_vec = model.predict(x=None, category=cat)
            weights.append(embedded_vec.squeeze(0).numpy())

    weights = torch.FloatTensor(weights)
    images = torch.FloatTensor(images)
    if show_image_on_board:
        writer.add_embedding(weights, label_img=images)
    else:
        writer.add_embedding(weights, metadata=labels)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=10, help='number of class')
    parser.add_argument('--resume-model', default='./results/model.pth', help='path to trained model')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--show-image-on-board', action='store_false')
    parser.add_argument('--show-all-embedding', action='store_false')
    parser.add_argument('--out-dir', default='./results', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
