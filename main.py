import torch

from config import SEED, EPOCHS, PATIENCE, OUTPUT_DIR
from utils import set_seed, log_environment
from dataset import build_dataloaders
from trainer import run_variant


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    log_environment(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders()

    run_variant(
        name="baseline_unet",
        use_dilation=False,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        patience=PATIENCE,
        seed=SEED,
    )

    run_variant(
        name="controlled_dilated_unet",
        use_dilation=True,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        patience=PATIENCE,
        seed=SEED,
    )


if __name__ == "__main__":
    main()