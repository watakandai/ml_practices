from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print('Episode:{}, {}'.format(episode, string))
        if step > 0:
            print('Step:{}, {}'.format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)

def _is_update(episode, freq):
    if episode!=0 and episode%freq==0:
        return True
    return False