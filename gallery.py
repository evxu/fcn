import os
from jinja2 import Environment, FileSystemLoader

TMPL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                './templates')
env = Environment(loader=FileSystemLoader(searchpath=TMPL_DIR))
tmpl = env.get_template('gallery.html')

class Gallery:
    def __init__ (self, path, ext = '.png', score=False):
        self.next_id = 0
        self.path = path
        self.ext = ext
        self.images = []
        self.score = score
        try:
            os.makedirs(path)
        except:
            pass
        pass

    def next (self, score=0):
        path = '%03d%s' % (self.next_id, self.ext)
        self.images.append((path, score))
        self.next_id += 1
        return os.path.join(self.path, path)

    def flush (self, rank=False):
        if rank:
            self.images = sorted(self.images, key=lambda x: -x[1])
        with open(os.path.join(self.path, 'index.html'), 'w') as f:
            f.write(tmpl.render(images = self.images, score=self.score))
            pass
        pass

