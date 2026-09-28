"""AR(2) entry point using the shared runner without changing pinned run code.

Example: python -m assignment3.ar2_experiments cv --output runs/cv_ar2
"""

from assignment3 import experiments


def main():
    experiments.DATASETS = ('ar2',)
    experiments.main()


if __name__ == '__main__':
    main()
