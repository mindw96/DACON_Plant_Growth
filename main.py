from torch import optim

from utils import seed_everything, LoadTrainData, LoadTestData, RMSELoss, train, test, draw, submit
from model import MyModel


def main(lr=0.0, epochs=0, batch_size=0):
    seed_everything(9608)

    train_dataset = LoadTrainData(batch_size=batch_size)
    train_loader = train_dataset.train_load()
    valid_loader = train_dataset.valid_load()

    test_dataset = LoadTestData(batch_size=batch_size)
    test_loader = test_dataset.test_load()

    model = MyModel()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RMSELoss()

    model, train_loss, valid_loss, best_idx = train(model=model, epochs=epochs, train_loader=train_loader,
                                                    valid_loader=valid_loader, optimizer=optimizer, criterion=criterion)

    draw(train_loss=train_loss, valid_loss=valid_loss, best_idx=best_idx)

    predict = test(model, test_loader)
    submit(predict)


if __name__ == '__main__':
    learning_rate = 1e-5
    epochs = 10
    batch_size = 128
    main(lr=learning_rate, epochs=epochs, batch_size=batch_size)


