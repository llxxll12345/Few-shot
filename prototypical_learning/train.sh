python3 download.py -i 1C5mdSZUmMYH5Qtd-KMuogA4qYa1seCXj -f images_background.tar.gz -d omniglot
python3 download.py -i 1CM72c7fsQBtpZzc90gvqKAsQWXRdUzpf -f images_evaluation.tar.gz -d omniglot
cd omniglot
tar -xzf images_background.tar.gz 
tar -xzf images_evaluation.tar.gz
# python3 train.py