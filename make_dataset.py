from dataset.scripts.make_interm import make_interm
from dataset.scripts.make_processed import make_processed

def main():
    make_interm(format = ('npy', 'img', 'meta'))
    make_interm(interpolate = True, format = ('npy', 'img', 'meta'))
    make_processed(format = ('npy', 'img', 'meta'))
    make_processed(interpolate = True, format = ('npy', 'img', 'meta'))

if __name__ == "__main__":
    main()
