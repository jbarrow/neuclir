from distutils.core import setup

setup(
    name='NeuCLIR',
    version='0.1dev',
    packages=['neuclir','neuclir.readers','neuclir.models','neuclir.datasets','neuclir.metrics', 'neuclir.predictors', 'neuclir.commands', 'neuclir.models.scorers'],
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
