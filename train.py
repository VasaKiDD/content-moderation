import glob
import os
import shutil
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import prepare_data_generator
from optimizer import SGDCosineAnnealed, ClassificationAccuracy, SpecialClassificationAccuracy
from utils import load_params, create_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_epoch(model, train_status, writer, test_loader):

    print("Testing the model...")

    model.eval()
    categories_accuracy = 0
    gender_accuracy = 0
    xxs_accuracy = 0
    total_non_nulls = 0

    acc = ClassificationAccuracy()
    spec_acc = SpecialClassificationAccuracy()

    for batch in tqdm(test_loader):
        input_images, target_categories, target_genders, target_xxs = batch
        pred_categories, pred_genders, pred_xxs = model(input_images.to(device))
        categories_accuracy += acc(pred_categories, target_categories.to(device))
        xxs_accuracy += acc(pred_xxs, target_xxs.to(device))
        gender_acc, non_nulls = spec_acc(pred_genders, target_genders.to(device))
        gender_accuracy += gender_acc
        total_non_nulls += non_nulls

    categories_accuracy /= test_loader.dataset.__len__()
    xxs_accuracy /= test_loader.dataset.__len__()
    gender_accuracy /= total_non_nulls

    writer.add_scalar(
        "Accuracy/test/0_to_5", categories_accuracy, train_status["batch"] * train_status["epoch"]
    )
    writer.add_scalar(
        "Accuracy/test/0_1_2+", xxs_accuracy, train_status["batch"] * train_status["epoch"]
    )
    writer.add_scalar(
        "Accuracy/test/male_woman_cartoon/",
        gender_accuracy,
        train_status["batch"] * train_status["epoch"],
    )

    return categories_accuracy, gender_accuracy, xxs_accuracy


def eval_epoch(model, train_status, writer, validation_loader):

    print("Evaluating...")

    model.eval()
    categories_accuracy = 0
    gender_accuracy = 0
    xxs_accuracy = 0
    total_non_nulls = 0

    acc = ClassificationAccuracy()
    spec_acc = SpecialClassificationAccuracy()

    for batch in tqdm(validation_loader):
        input_images, target_categories, target_genders, target_xxs = batch
        pred_categories, pred_genders, pred_xxs = model(input_images.to(device))
        categories_accuracy += acc(pred_categories, target_categories.to(device))
        xxs_accuracy += acc(pred_xxs, target_xxs.to(device))
        gender_acc, non_nulls = spec_acc(pred_genders, target_genders.to(device))
        gender_accuracy += gender_acc
        total_non_nulls += non_nulls

    categories_accuracy /= validation_loader.dataset.__len__()
    xxs_accuracy /= validation_loader.dataset.__len__()
    gender_accuracy /= total_non_nulls

    writer.add_scalar(
        "Accuracy/val/0_to_5/", categories_accuracy, train_status["batch"] * train_status["epoch"]
    )
    writer.add_scalar(
        "Accuracy/val/0_1_2+/", xxs_accuracy, train_status["batch"] * train_status["epoch"]
    )
    writer.add_scalar(
        "Accuracy/val/male_woman_cartoon/",
        gender_accuracy,
        train_status["batch"] * train_status["epoch"],
    )

    return categories_accuracy, gender_accuracy, xxs_accuracy


def train_epoch(model, train_status, writer, train_loader, validation_loader, optim, params):

    model.train()

    cross_entropy = nn.CrossEntropyLoss()
    log_softmax = nn.LogSoftmax(dim=-1)

    acc = ClassificationAccuracy()
    spec_acc = SpecialClassificationAccuracy()

    for batch in train_loader:
        input_images, target_categories, target_genders, target_xxs = batch

        pred_categories, pred_genders, pred_xxs = model(input_images.to(device))

        categories_accuracy = (
            acc(pred_categories, target_categories.to(device)) / params["training"]["batch_size"]
        )
        xxs_accuracy = acc(pred_xxs, target_xxs.to(device)) / params["training"]["batch_size"]
        gender_acc, non_nulls = spec_acc(pred_genders, target_genders.to(device))
        gender_accuracy = gender_acc / non_nulls

        category_loss = cross_entropy(pred_categories, target_categories.to(device))
        xxs_loss = cross_entropy(pred_xxs, target_xxs.to(device))
        gender_loss = (-1.0 * log_softmax(pred_genders) * target_genders.to(device)).mean()

        total_loss = 0.45 * category_loss + 0.275 * xxs_loss + 0.275 * gender_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        train_status["batch"] += 1

        print(
            "Epoch : {epoch}, Batch : {batch}".format(
                epoch=train_status["epoch"], batch=train_status["batch"]
            )
        )
        print("Loss/0_to_5/train :", category_loss)
        print("Loss/0_1_2+/train :", xxs_loss)
        print("Loss/male_woman_cartoon/train :", xxs_loss)
        print("Accuracy/0_to_5/train :", categories_accuracy)
        print("Accuracy/0_1_2+/train :", xxs_accuracy)
        print("Accuracy/male_woman_cartoon/train :", gender_accuracy)
        print("\n")

        writer.add_scalar(
            "Loss/train/0_to_5", category_loss, train_status["batch"] * train_status["epoch"]
        )
        writer.add_scalar(
            "Loss/train/0_1_2+", xxs_loss, train_status["batch"] * train_status["epoch"]
        )
        writer.add_scalar(
            "Loss/train/male_woman_cartoon",
            xxs_loss,
            train_status["batch"] * train_status["epoch"],
        )
        writer.add_scalar(
            "Accuracy/train/0_to_5",
            categories_accuracy,
            train_status["batch"] * train_status["epoch"],
        )
        writer.add_scalar(
            "Accuracy/train/0_1_2+", xxs_accuracy, train_status["batch"] * train_status["epoch"]
        )
        writer.add_scalar(
            "Accuracy/train/male_woman_cartoon",
            gender_accuracy,
            train_status["batch"] * train_status["epoch"],
        )

        if train_status["batch"] % params["training"]["eval_save_every"] == 0:
            categories_accuracy_val, gender_accuracy_val, xxs_accuracy_val = eval_epoch(
                model, train_status, writer, validation_loader
            )

            now = (
                str(datetime.now())
                .replace("-", "_")
                .replace(" ", "_")
                .replace(":", "_")
                .split(".")[0]
            )
            check_name = "{model}_{epoch}_{batch}_{date}.pth".format(
                model=params["model"],
                epoch=train_status["epoch"],
                batch=train_status["batch"],
                date=now,
            )
            with open(params["training"]["checkpoint_dir"] + "/training_stats.tsv", "a") as f:
                line = "{check_name}\t{epoch}\t{batch}\t{category_accuracy}\t{gender_accuracy}\t{xxs_accuracy}\n".format(
                    check_name=check_name,
                    epoch=train_status["epoch"],
                    batch=train_status["batch"],
                    category_accuracy=categories_accuracy_val,
                    gender_accuracy=gender_accuracy_val,
                    xxs_accuracy=xxs_accuracy_val,
                )
                f.write(line)

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "train_loader": train_loader,
                "epoch": train_status["epoch"],
                "batch": train_status["batch"],
            }

            torch.save(checkpoint, params["training"]["checkpoint_dir"] + "/" + check_name)


def train(model, train_status, train_loader, validation_loader, test_loader, optim, params):
    writer = SummaryWriter()

    for epoch_i in range(train_status["epoch"], params["training"]["epochs"]):

        train_epoch(model, train_status, writer, train_loader, validation_loader, optim, params)
        test_epoch(model, train_status, writer, test_loader)
        train_status["epoch"] = epoch_i + 1


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    params = load_params(args.parameters)

    train_loader, validation_loader, test_loader = prepare_data_generator(params)

    if "resnet" in params["model"]:
        import models.resnet

        model = getattr(models.resnet, params["model"])()
    elif "mnasnet" in params["model"]:
        import models.mnasnet

        model = getattr(models.mnasnet, params["model"])()
    else:
        print("Invalid model name in parameter file ! Exiting...")
        sys.exit()

    model.to(device)

    opti = SGDCosineAnnealed(
        model,
        batch_size=params["training"]["batch_size"],
        train_size=params["training"]["optimizer"]["period"],
        momentum=params["training"]["optimizer"]["momentum"],
        lmax=params["training"]["optimizer"]["lmax"],
        lmin=params["training"]["optimizer"]["lmin"],
        T0=params["training"]["optimizer"]["T0"],
        l2=params["training"]["optimizer"]["l2"],
        gradient_clip=params["training"]["optimizer"]["gradient_clip"],
    )

    if os.path.exists("runs/"):
        shutil.rmtree("runs")

    if os.path.exists(params["training"]["checkpoint_dir"]):
        if args.spec_checkpoint is None:
            checkpoints = glob.glob(params["training"]["checkpoint_dir"] + "/*")
            checkpoints.remove(params["training"]["checkpoint_dir"] + "/training_stats.tsv")
            checkpoints.sort()
            checkpoint = checkpoints[-1]
        else:
            checkpoint = args.spec_checkpoint
        checkpoint = torch.load(checkpoint)
        train_status = {}
        train_status["epoch"] = checkpoint["epoch"]
        train_status["batch"] = checkpoint["batch"]
        opti.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        train_loader = checkpoint["train_loader"]
        print(
            "Resuming with {model} at epoch {epoch} and batch {batch}...".format(
                model=params["model"],
                epoch=train_status["epoch"],
                batch=train_status["batch"],
            )
        )
    else:
        os.mkdir(params["training"]["checkpoint_dir"])
        with open(params["training"]["checkpoint_dir"] + "/training_stats.tsv", "w") as f:
            f.write(
                "directory\tmodel\tepoch\tbatch\tcategory_accuracy\tgender_accuracy\t0_1_2+_accuracy\n"
            )
        train_status = {}
        train_status["epoch"] = 1
        train_status["batch"] = 0
        print("Training from scratch {model}...".format(model=params["model"]))

    train(model, train_status, train_loader, validation_loader, test_loader, opti, params)
