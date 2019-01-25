import track
import skeletor
from src.data_handler import hello_world
from src.settings import GlobalSettings as GS
from src.utils import seed_all, print_settings

def add_args(parser):
    parser.add_argument('--lr', default=0.01, type=float)

def init(args):
    seed_all(args.seed)
    args.trial_path = track.trial_dir()
    GS.args = args
    print_settings()

def experiment(args):
    init(args)
    hello_world()

if __name__ == "__main__":
    skeletor.supply_args(add_args)
    skeletor.execute(experiment)
