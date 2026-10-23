"""Load result pickles written before the package was renamed from `cortisol` to `guarded_distillation`."""
import pickle


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "cortisol" or module.startswith("cortisol."):
            module = "guarded_distillation" + module[len("cortisol"):]
        return super().find_class(module, name)


def load_pickle(f):
    return _Unpickler(f).load()
