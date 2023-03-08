"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import time
from JoTools.txkjRes.deteRes import DeteRes
import numpy as np

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE   = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE          = "cuda"
BATCH_SIZE      = 12 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY    = 0
EPOCHS          = 1000
NUM_WORKERS     = 3
PIN_MEMORY      = True
LOAD_MODEL      = True
BEST_MAP        = -1
BEST_MODEL_FILE = r"/home/ldq/hand_yolo/best.pth"
LAST_MODEL_FILE = r"/home/ldq/hand_yolo/last.pth"

IMG_DIR     = r"/home/ldq/hand_yolo/data/images"
LABEL_DIR   = r"/home/ldq/hand_yolo/data/labels"
example_csv = r"/home/ldq/hand_yolo/data/train.csv"
test_csv    = r"/home/ldq/hand_yolo/data/test.csv"



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    
    BEST_MAP    = -1
    model       = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn     = YoloLoss()
    loss_fn.lambda_coord = 5
    loss_fn.lambda_noobj = 0.2  # 0.5 

    if LOAD_MODEL:
        load_checkpoint(torch.load(BEST_MODEL_FILE), model, optimizer)

    train_dataset   = VOCDataset(example_csv, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,)
    test_dataset    = VOCDataset(test_csv, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,)
    train_loader    = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True,)
    test_loader     = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True,)

    for epoch in range(EPOCHS):
        
        # # draw res for check
        # for x, y in test_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(5):
        #         dete_res = DeteRes()
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #         img = np.array(x[idx].permute(1,2,0).to("cpu"))
        #         img = (img*255)
        #         img = img.astype(np.uint8).copy()

        #         height, width, _ = img.shape
        #         dete_res.img_ndarry = img

        #         for each_box in bboxes:
        #             box = each_box[2:]
        #             assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        #             upper_left_x = box[0] - box[2] / 2
        #             upper_left_y = box[1] - box[3] / 2
        #             down_right_x = box[0] + box[2] / 2
        #             down_right_y = box[1] + box[3] / 2
        #             # 
        #             x1 = int(max(upper_left_x * width, 1))
        #             y1 = int(max(upper_left_y * height, 1))
        #             x2 = int(min(down_right_x * width, width-1))
        #             y2 = int(min(down_right_y * height, height-1))
        #             dete_res.add_obj(x1 = x1, y1=y1, x2=x2,y2=y2, tag=str(each_box[0]), conf=float(each_box[1]))

        #         dete_res.print_as_fzc_format()
        #         dete_res.draw_dete_res(f"./draw/{idx}.jpg")

        #     exit()


        # get pred_box
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
        
        # train
        train_fn(train_loader, model, optimizer, loss_fn)

        # get map
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"{epoch} : Train mAP: {mean_avg_prec}")

        if mean_avg_prec > BEST_MAP:
            BEST_MAP = mean_avg_prec
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
            save_checkpoint(checkpoint, filename=BEST_MODEL_FILE)
            time.sleep(10)
        
        # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
        # save_checkpoint(checkpoint, filename=LAST_MODEL_FILE)
        # time.sleep(10)
        


if __name__ == "__main__":

    main()
