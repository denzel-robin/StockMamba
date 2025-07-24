# StockMamba

## Recommended to run in colab or in an isolated python environment
- This requirements causes conflict in both colab and local

1) Setup the environment
2) Install the accurate versions of modules(required for colab and local computer)

``` shell
pip install torch==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja einops packaging
```

``` shell
git clone https://github.com/state-spaces/mamba.git
cd mamba
```

```shell
pip install -e
```

