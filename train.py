"""Train and evaluate the bike-demand neural network."""

import copy

import torch
from torch import nn

from bikes_data import make_rentals
from model import BikeDemandNetwork
from prepare import SEED, make_loaders, prepare_data


def run_epoch(model, loader, loss_function, optimizer=None):
    """Train for one epoch, or measure when no optimizer is supplied."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_examples = 0

    for features, targets in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = loss_function(predictions, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(predictions - targets).sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples
    mean_absolute_error = total_absolute_error / total_examples

    return mean_loss, mean_absolute_error


def train_model(
    model,
    training_loader,
    validation_loader,
    epochs=100,
    learning_rate=0.001,
):
    """Train for several epochs and restore the best validation weights."""

    if epochs < 1:
        raise ValueError("Training requires at least one epoch.")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        training_loss, training_mae = run_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
        )

        validation_loss, validation_mae = run_epoch(
            model,
            validation_loader,
            loss_function,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_mae": training_mae,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"training loss: {training_loss:.2f} | "
            f"validation loss: {validation_loss:.2f} | "
            f"validation MAE: {validation_mae:.2f}"
        )

    model.load_state_dict(best_state)

    return history, best_epoch


def prove_small_batch_learning(steps=1000):
    """Confirm that the network can memorize one small batch."""

    torch.manual_seed(SEED)

    prepared = prepare_data(make_rentals())
    training_loader, _, _ = make_loaders(
        prepared,
        batch_size=16,
        seed=SEED,
    )

    features, targets = next(iter(training_loader))

    model = BikeDemandNetwork(input_size=features.shape[1])
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = loss_function(model(features), targets).item()

    for _ in range(steps):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

    final_loss = loss_function(model(features), targets).item()

    return initial_loss, final_loss


if __name__ == "__main__":
    starting_loss, ending_loss = prove_small_batch_learning()

    print(f"Initial small-batch loss: {starting_loss:.4f}")
    print(f"Final small-batch loss: {ending_loss:.4f}")
    print(f"Remaining fraction: {ending_loss / starting_loss:.6f}")
