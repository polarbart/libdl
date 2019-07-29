import pylibdl as libdl


class no_grad:

    def __enter__(self):
        libdl.set_no_grad(True)

    def __exit__(self, *args):
        libdl.set_no_grad(False)
