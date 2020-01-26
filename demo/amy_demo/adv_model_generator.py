import tensorflow as tf

if __name__ == '__main__':
    tf.enable_eager_execution()

from evaluate import *


def adversarial_explanation_wrapper(X_train, Y_train, model_full, z_idx, e_alpha=0.25, verbose=0):
    train_dataset = get_dataset(X_train, Y_train, batch_size=1000)
    model_explain = clone_model(model_full)
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    for t in range(50):
        train_err, train_loss = epoch_explanation(train_dataset, model_explain, attack=no_attack,
                                                  sensitive_feature_id=z_idx, e_alpha=e_alpha,
                                                  optimizer=optimizer)

    return model_explain


def evaluate_adversarial_explanation(X_train, Y_train, X_test, Y_test, model_explain, z_idx):
    raise NotImplementedError
    train_dataset = get_dataset(X_train, Y_train, batch_size=1000)
    test_dataset = get_dataset(X_test, Y_test, batch_size=1000)
    adv_err, adv_err_f, e_loss, e_loss_train, test_err = epoch_eval(train_dataset, test_dataset, model_explain, z_idx)
    r = (train_err, test_err, adv_err, adv_err_f, e_loss, e_loss_train)


def main():
    pass


if __name__ == '__main__':
    main()
