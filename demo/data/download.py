import gdown





def main():
    url = "https://drive.google.com/uc?id=1HfcB_U5dZmnQbBL2OFvpiusm9nA1y3FI"
    output = 'german.data-numeric'
    gdown.download(url, output, quiet=False)


    url = "https://drive.google.com/uc?id=1H_bOI2dAuZsAcRrtMmQSR65aOxKldtgA"
    output = "adult.npz"
    gdown.download(url, output, quiet=False)

    url = "https://drive.google.com/uc?id=16tR-tML2XXBSQWsvOMtU7zvVakQrixW4"
    output = "bank.npz"
    gdown.download(url, output, quiet=False)

    url = "https://drive.google.com/uc?id=1wZsk12TC_ggCF9emdp-kc-gG-99t9oeT"
    output = "compas.npy"
    gdown.download(url, output, quiet=False)


if __name__ == '__main__':
    main()