from distutils.core import setup

setup(
    name='NeuCLIR',
    version='0.1dev',
    packages=[
        'neuclir',
        'neuclir.commands',
        'neuclir.config',
        'neuclir.data',
        'neuclir.data.dataset_readers',
        'neuclir.data.iterators',
        'neuclir.data.minglers',
        'neuclir.losses',
        'neuclir.metrics',
        'neuclir.models', 'neuclir.models.scorers',
        'neuclir.predictors',
        'neuclir.readers',
        'neuclir.training', 'neuclir.training'],
    license='Academic License',
    install_requires=[
        'allennlp>=0.7.1'
    ],
    long_description=open('README.md').read(),
    entry_points={
        'console_scripts': [
            "neuclir=neuclir.run:run"
        ]
    },
)
