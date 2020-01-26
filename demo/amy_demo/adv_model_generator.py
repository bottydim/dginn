import tensorflow as tf

if __name__=='__main__':
    tf.enable_eager_execution()



def get_adversarial_model(X_train, Y_train, X_test, Y_test, e_alpha=0.25, feature_set=None, EPOCHS=1000, R_EPOCHS=100, batch_size=30000,
                              seed=49, num_layers=3):

    models_p = []
    adv_lis = []
    val_names = ("train_err", "test_err", "adv_err", "adv_err_fgsm", "e_loss", "e_loss_train")
    print(*("{}".format(i) for i in val_names), sep="\t")
    if feature_set is None:
        feature_set = range(n_features)
    for i in feature_set:
        print(i)
        z_idx = i
        model_explain = clone_model(model_full)
        optimizer = tf.keras.optimizers.Adam(lr=0.01)
        for t in range(50):
            train_err, train_loss = epoch_explanation(adult_train, model_explain, attack,
                                                      sensitive_feature_id=z_idx, e_alpha=e_alpha,
                                                      epsilon=0.25, alpha=0.08, num_iter=30, optimizer=optimizer)
        models_p.append(model_explain)
        adv_err, adv_err_f, e_loss, e_loss_train, test_err = epoch_eval(adult_train, adult_test, model_explain, z_idx)
        r = (train_err, test_err, adv_err, adv_err_f, e_loss, e_loss_train)
        adv_lis.append(r)
        print(*("{:.6f}".format(i) for i in r), sep="\t")
