import click
from dataset.scripts.make_interm import make_interm
from dataset.scripts.make_processed import make_processed

@click.command()
@click.option('--full', '-f', is_flag=True, default=False, 
              help="If set, inversions will be generated")

def main(full:bool = False):
    if full:
        make_interm(format = ('npy', 'img', 'meta', 'inv'))
        make_interm(interpolate = True, format = ('npy', 'img', 'meta', 'inv'))
        make_processed(format = ('npy', 'img', 'meta', 'inv'))
        make_processed(interpolate = True, format = ('npy', 'img', 'meta', 'inv'))
    else:
        make_interm(format = ('npy', 'img', 'meta'))
        make_interm(interpolate = True, format = ('npy', 'img', 'meta'))
        make_processed(format = ('npy', 'img', 'meta'))
        make_processed(interpolate = True, format = ('npy', 'img', 'meta'))

if __name__ == "__main__":
    main()
