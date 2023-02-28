from tqdm import tqdm

class ProgressBar(object):
    def __init__(self, disable=False):
        self.disable = disable
        self.count = 0
        self.max_value = 0
        self.p = None
    
    def pbar(self, num, total_num, max_value):
        # Re-assign value
        self.count = 0
        self.max_value = max_value

        self.p = tqdm(
            total=max_value,
            desc=f'Epoch: {num}/{total_num} ',
            disable=self.disable,
            colour='#0216ad',
            leave=True
        )
        # return self.p

    def update(self, *args, update_value=1):
        self.p.update(update_value)
        self.p.set_postfix(loss = args[0], acc = args[1])
        self.count = self.count + 1

    def close(self):
        self.p.colour = "#018005"
        self.p.close()

    def failed(self):
        self.p.colour = "red"
        self.p.close()