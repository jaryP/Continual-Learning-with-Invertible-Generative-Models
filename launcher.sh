for d in $(find . -wholename './experiments/files/mnist/*.yaml') ; do
    python main.py "$d" --cuda 0
done
python main.py '/experiments/files/mnist/naive.yaml' --cuda 0
